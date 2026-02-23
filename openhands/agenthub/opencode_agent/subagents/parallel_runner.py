"""Parallel subagent execution engine.

Runs multiple subagent loops concurrently using ThreadPoolExecutor.
Each subagent gets its own thread, agent instance, and local state.
Action execution goes directly to the runtime HTTP endpoint, bypassing
the event stream to avoid recursive on_event calls and event interleaving.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import httpx

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    MessageAction,
)
from openhands.events.action.agent import ValidationFailureAction
from openhands.events.event import EventSource
from openhands.events.observation.error import ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.events.serialization.observation import observation_from_dict
from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from openhands.core.config import AgentConfig


MAX_SUBAGENT_STEPS = 50


class ParallelSubagentRunner:
    """Runs multiple subagent loops concurrently.

    Each subagent is spawned in a separate thread. Tool actions are sent
    to the runtime HTTP server directly (bypassing the event stream) so
    multiple subagents can execute independently without interfering with
    the parent controller's event-driven flow.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        parent_agent_config: 'AgentConfig',
        agent_configs: dict[str, 'AgentConfig'],
        runtime_url: str | None = None,
    ):
        self.llm_registry = llm_registry
        self.parent_agent_config = parent_agent_config
        self.agent_configs = agent_configs
        self.runtime_url = runtime_url or os.environ.get('OPENHANDS_RUNTIME_URL')

    def run_parallel(
        self, delegate_actions: list[AgentDelegateAction]
    ) -> dict[str, dict[str, Any]]:
        """Run subagent tasks in parallel.

        Args:
            delegate_actions: List of AgentDelegateAction, each representing
                a subagent task. Must have tool_call_metadata set.

        Returns:
            Dict mapping tool_call_id -> result dict with 'content' and 'outputs'.
        """
        if not self.runtime_url:
            raise RuntimeError(
                'Cannot run parallel subagents: OPENHANDS_RUNTIME_URL not set. '
                'The runtime must be started before parallel execution.'
            )

        logger.info(
            f'Starting parallel execution of {len(delegate_actions)} subagents '
            f'(runtime: {self.runtime_url})'
        )
        start_time = time.monotonic()

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(delegate_actions)) as executor:
            futures = {
                executor.submit(self._run_single, action): action
                for action in delegate_actions
            }
            for future in as_completed(futures):
                action = futures[future]
                tool_call_id = action.tool_call_metadata.tool_call_id
                try:
                    result = future.result()
                    results[tool_call_id] = result
                except Exception as e:
                    logger.error(
                        f'Subagent {action.agent} failed: {e}', exc_info=True
                    )
                    results[tool_call_id] = {
                        'content': f'Subagent error: {e}',
                        'outputs': {},
                    }

        elapsed = time.monotonic() - start_time
        logger.info(
            f'Parallel subagent execution completed in {elapsed:.1f}s '
            f'({len(results)} results)'
        )
        return results

    def _run_single(self, delegate_action: AgentDelegateAction) -> dict[str, Any]:
        """Run a single subagent loop to completion in the current thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return self._run_subagent_loop(delegate_action)
        finally:
            loop.close()

    def _run_subagent_loop(
        self, delegate_action: AgentDelegateAction
    ) -> dict[str, Any]:
        agent_name = delegate_action.agent
        inputs = delegate_action.inputs

        agent_cls = Agent.get_cls(agent_name)
        agent_config = self.agent_configs.get(agent_name, self.parent_agent_config)
        agent = agent_cls(config=agent_config, llm_registry=self.llm_registry)

        state = self._create_subagent_state(agent, inputs)
        id_counter = len(state.history)

        step_count = 0
        while step_count < MAX_SUBAGENT_STEPS:
            step_count += 1
            try:
                action = agent.step(state)
            except Exception as e:
                logger.error(f'Subagent {agent_name} step failed: {e}', exc_info=True)
                return {
                    'content': f'Subagent step error after {step_count} steps: {e}',
                    'outputs': {},
                }

            if action is None:
                continue

            action._id = id_counter
            id_counter += 1
            action._source = EventSource.AGENT

            if isinstance(action, AgentFinishAction):
                outputs = action.outputs or {}
                content = outputs.get('content', action.final_thought or '')
                return {'content': content, 'outputs': outputs}

            if isinstance(action, (AgentThinkAction, ValidationFailureAction)):
                state.history.append(action)
                continue

            if action.runnable:
                obs = self._execute_action(action)
                obs._cause = action.id
                obs._id = id_counter
                id_counter += 1
                if hasattr(action, 'tool_call_metadata') and action.tool_call_metadata:
                    obs.tool_call_metadata = action.tool_call_metadata
                state.history.append(action)
                state.history.append(obs)
            else:
                state.history.append(action)

        return {
            'content': f'Subagent {agent_name} reached max steps ({MAX_SUBAGENT_STEPS})',
            'outputs': {},
        }

    def _create_subagent_state(self, agent: Agent, inputs: dict) -> State:
        """Create a minimal local State for a subagent."""
        state = State(
            session_id=f'parallel-subagent-{id(agent)}',
            inputs=inputs,
        )
        state.agent_state = 'running'

        system_msg = agent.get_system_message()
        if system_msg:
            system_msg._id = 0
            state.history.append(system_msg)

        task_text = inputs.get('task', '')
        task_msg = MessageAction(content='TASK: ' + task_text)
        task_msg._source = EventSource.USER
        task_msg._id = 1 if system_msg else 0
        state.history.append(task_msg)

        return state

    def _execute_action(self, action: 'Action') -> 'Observation':
        """Execute an action via the runtime HTTP endpoint."""
        action_dict = event_to_dict(action)
        timeout = getattr(action, 'timeout', None) or 120

        try:
            with httpx.Client(timeout=timeout + 10) as client:
                response = client.post(
                    f'{self.runtime_url}/execute_action',
                    json={'action': action_dict},
                    timeout=timeout + 5,
                )
                response.raise_for_status()
                return observation_from_dict(response.json())
        except httpx.TimeoutException:
            return ErrorObservation(
                content=f'Action timed out after {timeout}s'
            )
        except Exception as e:
            return ErrorObservation(content=f'Action execution failed: {e}')
