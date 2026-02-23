import os
import sys
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.opencode_agent.function_calling as opencode_function_calling
from openhands.agenthub.opencode_agent.tools.apply_patch import ApplyPatchTool
from openhands.agenthub.opencode_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.opencode_agent.tools.edit import EditTool
from openhands.agenthub.opencode_agent.tools.finish import FinishTool
from openhands.agenthub.opencode_agent.tools.glob import GlobTool
from openhands.agenthub.opencode_agent.tools.grep import GrepTool
from openhands.agenthub.opencode_agent.tools.list_dir import ListDirTool
from openhands.agenthub.opencode_agent.tools.question import QuestionTool
from openhands.agenthub.opencode_agent.tools.read import ReadTool
from openhands.agenthub.opencode_agent.tools.task import TaskTool
from openhands.agenthub.opencode_agent.tools.think import ThinkTool
from openhands.agenthub.opencode_agent.tools.todo import TodoReadTool, TodoWriteTool
from openhands.agenthub.opencode_agent.tools.write import WriteTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import AgentDelegateAction, AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class OpenCodeAgent(Agent):
    VERSION = "1.0"
    """
    The OpenCode Agent is inspired by Claude Code's tool-based approach.

    This agent provides a comprehensive set of tools for file operations, web access,
    user interaction, and task management. It emphasizes structured workflows and
    clear tool boundaries.

    ### Key Features

    1. **File Operations**: Read, write, edit, glob, grep, list_dir
    2. **User Interaction**: question tool for clarifying requirements
    3. **Task Management**: todo_read, todo_write for tracking progress
    4. **Code Execution**: bash commands for running scripts and tools
    5. **Structured Edits**: apply_patch for multi-file changes

    The agent uses function calling to invoke tools and maintains conversation
    history for context-aware assistance.
    """

    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the OpenCodeAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque["Action"] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f"Using condenser: {type(self.condenser)}")

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            # Use custom prompt directory if configured, otherwise use default
            prompt_dir = (
                self.config.custom_prompt_dir
                if self.config.custom_prompt_dir
                else os.path.join(os.path.dirname(__file__), "prompts")
            )

            # Build template overrides from custom paths
            template_overrides = {}
            if self.config.system_prompt_path:
                template_overrides["system_prompt.j2"] = self.config.system_prompt_path
            if self.config.system_prompt_long_horizon_path:
                template_overrides["system_prompt_long_horizon.j2"] = (
                    self.config.system_prompt_long_horizon_path
                )

            self._prompt_manager = PromptManager(
                prompt_dir=prompt_dir,
                system_prompt_filename=self.config.resolved_system_prompt_filename,
                template_overrides=template_overrides if template_overrides else None,
            )

        return self._prompt_manager

    def _get_tools(self) -> list["ChatCompletionToolParam"]:
        """Get the list of tools available to the OpenCode agent."""
        tools = []

        # Core file operation tools (OpenCode-style)
        tools.append(ReadTool)
        tools.append(WriteTool)
        tools.append(EditTool)

        # File search and navigation tools
        tools.append(GlobTool)
        tools.append(GrepTool)
        tools.append(ListDirTool)

        # User interaction tools
        # tools.append(QuestionTool)

        # Task management tools
        tools.append(TodoReadTool)
        tools.append(TodoWriteTool)

        # Subagent delegation
        tools.append(TaskTool)

        # Structured editing
        # tools.append(ApplyPatchTool)

        # Command execution
        if self.config.enable_cmd:
            use_short_desc = any(
                substr in self.llm.config.model
                for substr in ["gpt-4", "o3", "o1", "o4"]
            )
            tools.append(create_cmd_run_tool(use_short_description=use_short_desc))

        if self.config.enable_finish:
            tools.append(FinishTool)

        return tools

    def reset(self) -> None:
        """Resets the OpenCode Agent's internal state."""
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> "Action":
        """Performs one step using the OpenCode Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        When the LLM returns multiple task (subagent) tool calls and NO other tool
        calls, they are executed in parallel via ``ParallelSubagentRunner``.  The
        results are injected as synthetic events into state.history and the LLM is
        re-called so the agent sees all subagent outputs at once.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - Various Action types based on tool calls
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == "/exit":
            return AgentFinishAction()

        # Condense the events from the state
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f"Processing {len(condensed_history)} events from a total of {len(state.history)} events"
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            "messages": messages,
        }
        params["tools"] = check_tools(self.tools, self.llm.config)
        params["extra_body"] = {
            "metadata": state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        response = self.llm.completion(**params)
        logger.debug(f"Response from LLM: {response}")
        actions = self.response_to_actions(response)
        logger.debug(f"Actions after response_to_actions: {actions}")

        # --- Parallel subagent execution ---
        # When ALL actions in the batch are AgentDelegateAction (task tool calls)
        # and there are ≥2, run them concurrently instead of sequentially.
        delegate_actions = [a for a in actions if isinstance(a, AgentDelegateAction)]
        if (
            len(delegate_actions) >= 2
            and len(delegate_actions) == len(actions)
            and os.environ.get('OPENHANDS_RUNTIME_URL')
        ):
            return self._handle_parallel_subagents(
                delegate_actions, state, response, messages, params
            )

        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _handle_parallel_subagents(
        self,
        delegate_actions: list[AgentDelegateAction],
        state: State,
        original_response: "ModelResponse",
        messages: list[Message],
        params: dict,
    ) -> "Action":
        """Execute multiple subagent task calls in parallel, inject results, re-call LLM.

        This method mirrors how opencode handles concurrent task tool calls:
        all tasks run simultaneously, their results are returned as tool results,
        and the LLM generates its next response based on the combined outputs.
        """
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        logger.info(
            f'Parallel subagent execution: {len(delegate_actions)} task calls detected'
        )

        runner = ParallelSubagentRunner(
            llm_registry=self.llm_registry,
            parent_agent_config=self.config,
            agent_configs={},
        )

        results = runner.run_parallel(delegate_actions)

        # Inject synthetic events into state.history so the conversation memory
        # picks them up correctly on subsequent calls to _get_messages().
        base_id = 10_000_000 + len(state.history) * 100
        for i, delegate_action in enumerate(delegate_actions):
            tc_id = delegate_action.tool_call_metadata.tool_call_id
            result = results.get(tc_id, {'content': 'No result', 'outputs': {}})

            synth_action = AgentDelegateAction(
                agent=delegate_action.agent,
                inputs=delegate_action.inputs,
            )
            synth_action._id = base_id + i * 2
            synth_action.tool_call_metadata = delegate_action.tool_call_metadata

            synth_obs = AgentDelegateObservation(
                outputs=result.get('outputs', {}),
                content=result.get('content', ''),
            )
            synth_obs._id = base_id + i * 2 + 1
            synth_obs._cause = synth_action._id
            synth_obs.tool_call_metadata = delegate_action.tool_call_metadata

            state.history.append(synth_action)
            state.history.append(synth_obs)

        # Re-build messages including the synthetic events and re-call LLM
        condensed_history_new: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history_new = events
            case Condensation(action=condensation_action):
                return condensation_action

        initial_user_message = self._get_initial_user_message(state.history)
        new_messages = self._get_messages(condensed_history_new, initial_user_message)
        params["messages"] = new_messages
        new_response = self.llm.completion(**params)
        new_actions = self.response_to_actions(new_response)

        for action in new_actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Finds the initial user message action from the full history."""
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == "user":
                initial_user_message = event
                break

        if initial_user_message is None:
            logger.error(
                f"CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history."
            )
            raise ValueError(
                "Initial user message not found in history. Please report this issue."
            )
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation."""
        if not self.prompt_manager:
            raise Exception("Prompt Manager not instantiated.")

        # Use ConversationMemory to process events
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def response_to_actions(self, response: "ModelResponse") -> list["Action"]:
        return opencode_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
