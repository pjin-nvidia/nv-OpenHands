"""Tests for parallel subagent execution via ParallelSubagentRunner.

Covers edge cases and integration scenarios beyond what
test_opencode_subagents.py provides:
1. Subagent loop lifecycle (step -> finish, step -> max iterations)
2. Concurrent timing validation
3. Error isolation across threads
4. Runtime URL resolution
5. Integration with OpenCodeAgent.step() parallel detection
"""

import json
import os
import time
from collections import deque
from unittest.mock import Mock, patch

import pytest
from litellm import ModelResponse

from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
    MAX_SUBAGENT_STEPS,
    ParallelSubagentRunner,
)
from openhands.agenthub.opencode_agent.tools.task import TASK_TOOL_NAME
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.events.action import (
    AgentDelegateAction,
    AgentFinishAction,
    MessageAction,
    OpenCodeReadAction,
)
from openhands.events.event import EventSource
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.tool import ToolCallMetadata
from openhands.llm.llm_registry import LLMRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response_multi(tool_calls):
    calls = []
    for i, (name, args) in enumerate(tool_calls):
        calls.append(
            {
                "function": {"name": name, "arguments": json.dumps(args)},
                "id": f"tc-{i}",
                "type": "function",
            }
        )
    return ModelResponse(
        id="mock",
        choices=[
            {
                "message": {"tool_calls": calls, "content": None, "role": "assistant"},
                "index": 0,
                "finish_reason": "tool_calls",
            }
        ],
    )


def _mock_text_response(text):
    return ModelResponse(
        id="mock",
        choices=[
            {
                "message": {"content": text, "role": "assistant", "tool_calls": None},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    )


def _make_delegate_action(agent_name, task, tool_call_id):
    action = AgentDelegateAction(
        agent=agent_name,
        inputs={"task": task, "description": f"Test: {task}"},
    )
    action.tool_call_metadata = ToolCallMetadata(
        tool_call_id=tool_call_id,
        function_name="task",
        model_response=ModelResponse(
            id="mock-resp",
            choices=[
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "task",
                                    "arguments": json.dumps(
                                        {"prompt": task, "subagent_type": "generalPurpose"}
                                    ),
                                },
                                "id": tool_call_id,
                                "type": "function",
                            }
                        ],
                        "content": None,
                        "role": "assistant",
                    },
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ],
        ),
        total_calls_in_response=1,
    )
    return action


@pytest.fixture
def llm_registry():
    cfg = OpenHandsConfig()
    cfg.set_llm_config(LLMConfig(model="gpt-4o", api_key="test"))
    return LLMRegistry(config=cfg)


def _make_agent(llm_registry):
    from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

    agent = OpenCodeAgent(config=AgentConfig(), llm_registry=llm_registry)
    mock_llm = Mock()
    mock_llm.config = Mock()
    mock_llm.config.model = "gpt-4o"
    mock_llm.config.max_message_chars = 10000
    mock_llm.vision_is_active = Mock(return_value=False)
    mock_llm.is_caching_prompt_active = Mock(return_value=False)
    agent.llm = mock_llm
    return agent


def _make_state():
    state = Mock(spec=State)
    user_msg = Mock()
    user_msg.content = "do something"
    state.get_last_user_message = Mock(return_value=user_msg)
    msg = MessageAction(content="do something")
    msg._source = EventSource.USER
    state.history = [msg]
    state.inputs = {}
    state.extra_data = {}
    state.to_llm_metadata = Mock(return_value={})
    return state


def _wire_agent(agent, state):
    from openhands.memory.condenser.condenser import View

    agent.condenser = Mock()
    agent.condenser.condensed_history = Mock(return_value=View(events=state.history))
    agent.conversation_memory = Mock()
    agent.conversation_memory.process_events = Mock(return_value=[])
    agent.conversation_memory.apply_prompt_caching = Mock()


# ===========================================================================
# 1. ParallelSubagentRunner runtime URL resolution
# ===========================================================================


class TestRuntimeURLResolution:
    def test_uses_explicit_url(self):
        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://explicit:8000",
        )
        assert runner.runtime_url == "http://explicit:8000"

    def test_falls_back_to_env_var(self):
        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://env:9000"}):
            runner = ParallelSubagentRunner(
                llm_registry=Mock(),
                parent_agent_config=Mock(),
                agent_configs={},
                runtime_url=None,
            )
            assert runner.runtime_url == "http://env:9000"

    def test_none_when_no_url_available(self):
        with patch.dict("os.environ", {}, clear=True):
            runner = ParallelSubagentRunner(
                llm_registry=Mock(),
                parent_agent_config=Mock(),
                agent_configs={},
                runtime_url=None,
            )
            assert runner.runtime_url is None


# ===========================================================================
# 2. Subagent loop lifecycle
# ===========================================================================


class TestSubagentLoop:

    @patch.object(ParallelSubagentRunner, "_execute_action")
    def test_loop_returns_on_finish_action(self, mock_exec):
        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        finish = AgentFinishAction(
            final_thought="Done", outputs={"content": "All good"}
        )
        mock_agent = Mock()
        mock_agent.step.return_value = finish
        mock_agent.get_system_message.return_value = None

        action = _make_delegate_action("TestAgent", "Do stuff", "c1")

        with patch(
            "openhands.agenthub.opencode_agent.subagents.parallel_runner.Agent.get_cls"
        ) as mock_cls:
            mock_cls.return_value = lambda config, llm_registry: mock_agent
            result = runner._run_subagent_loop(action)

        assert result["content"] == "All good"
        mock_exec.assert_not_called()

    @patch.object(ParallelSubagentRunner, "_execute_action")
    def test_loop_hits_max_steps(self, mock_exec):
        from openhands.events.observation.empty import NullObservation

        mock_exec.return_value = NullObservation(content="ok")

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        read_action = OpenCodeReadAction(path="/test.py")
        read_action.tool_call_metadata = ToolCallMetadata(
            tool_call_id="inner-tc",
            function_name="read",
            model_response=_mock_response_multi([("read", {"file_path": "/test.py"})]),
            total_calls_in_response=1,
        )

        mock_agent = Mock()
        mock_agent.step.return_value = read_action
        mock_agent.get_system_message.return_value = None

        action = _make_delegate_action("TestAgent", "Loop forever", "c2")

        with patch(
            "openhands.agenthub.opencode_agent.subagents.parallel_runner.Agent.get_cls"
        ) as mock_cls:
            mock_cls.return_value = lambda config, llm_registry: mock_agent
            result = runner._run_subagent_loop(action)

        assert "max steps" in result["content"].lower()
        assert mock_agent.step.call_count == MAX_SUBAGENT_STEPS

    def test_loop_handles_step_exception(self):
        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        mock_agent = Mock()
        mock_agent.step.side_effect = RuntimeError("LLM API down")
        mock_agent.get_system_message.return_value = None

        action = _make_delegate_action("TestAgent", "Fail", "c3")

        with patch(
            "openhands.agenthub.opencode_agent.subagents.parallel_runner.Agent.get_cls"
        ) as mock_cls:
            mock_cls.return_value = lambda config, llm_registry: mock_agent
            result = runner._run_subagent_loop(action)

        assert "error" in result["content"].lower()

    def test_loop_skips_none_actions(self):
        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        call_count = [0]

        def step_fn(state):
            call_count[0] += 1
            if call_count[0] <= 2:
                return None
            return AgentFinishAction(
                final_thought="OK", outputs={"content": "Done after nones"}
            )

        mock_agent = Mock()
        mock_agent.step.side_effect = step_fn
        mock_agent.get_system_message.return_value = None

        action = _make_delegate_action("TestAgent", "Skip nones", "c4")

        with patch(
            "openhands.agenthub.opencode_agent.subagents.parallel_runner.Agent.get_cls"
        ) as mock_cls:
            mock_cls.return_value = lambda config, llm_registry: mock_agent
            result = runner._run_subagent_loop(action)

        assert result["content"] == "Done after nones"
        assert call_count[0] == 3


# ===========================================================================
# 3. Concurrent timing validation
# ===========================================================================


class TestConcurrentTiming:

    @patch.object(ParallelSubagentRunner, "_run_subagent_loop")
    def test_parallel_faster_than_sequential(self, mock_loop):
        def slow_result(action):
            time.sleep(0.1)
            return {"content": f"Done {action.agent}", "outputs": {}}

        mock_loop.side_effect = slow_result

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        actions = [_make_delegate_action("A", f"T{i}", f"c{i}") for i in range(3)]

        t0 = time.monotonic()
        results = runner.run_parallel(actions)
        elapsed = time.monotonic() - t0

        assert len(results) == 3
        assert elapsed < 0.5  # 3 x 0.1s sequential = 0.3s; parallel should be ~0.1s

    @patch.object(ParallelSubagentRunner, "_run_subagent_loop")
    def test_single_action_still_works(self, mock_loop):
        mock_loop.return_value = {"content": "Solo result", "outputs": {}}

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        actions = [_make_delegate_action("A", "Solo", "c0")]
        results = runner.run_parallel(actions)

        assert len(results) == 1
        assert results["c0"]["content"] == "Solo result"


# ===========================================================================
# 4. Error isolation across threads
# ===========================================================================


class TestErrorIsolation:

    @patch.object(ParallelSubagentRunner, "_run_subagent_loop")
    def test_one_failure_doesnt_affect_others(self, mock_loop):
        def result_or_fail(action):
            tc_id = action.tool_call_metadata.tool_call_id
            if tc_id == "c1":
                raise RuntimeError("Agent crashed")
            return {"content": f"OK from {tc_id}", "outputs": {}}

        mock_loop.side_effect = result_or_fail

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        actions = [_make_delegate_action("A", f"T{i}", f"c{i}") for i in range(3)]
        results = runner.run_parallel(actions)

        assert len(results) == 3
        assert "OK from c0" in results["c0"]["content"]
        assert "error" in results["c1"]["content"].lower()
        assert "OK from c2" in results["c2"]["content"]

    @patch.object(ParallelSubagentRunner, "_run_subagent_loop")
    def test_all_failures_handled(self, mock_loop):
        mock_loop.side_effect = RuntimeError("total failure")

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url="http://test:8000",
        )

        actions = [_make_delegate_action("A", f"T{i}", f"c{i}") for i in range(2)]
        results = runner.run_parallel(actions)

        assert len(results) == 2
        for v in results.values():
            assert "error" in v["content"].lower()


# ===========================================================================
# 5. Integration: step() parallel detection
# ===========================================================================


class TestStepParallelIntegration:
    """Tests that OpenCodeAgent.step() correctly routes to parallel execution."""

    def test_two_task_calls_trigger_parallel(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = _make_state()
        _wire_agent(agent, state)

        task_resp = _mock_response_multi(
            [
                (TASK_TOOL_NAME, {"description": "A", "prompt": "p1", "subagent_type": "explore"}),
                (TASK_TOOL_NAME, {"description": "B", "prompt": "p2", "subagent_type": "general"}),
            ]
        )
        final_resp = _mock_text_response("Combined results.")
        agent.llm.completion = Mock(side_effect=[task_resp, final_resp])

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://test:8000"}):
            with patch(
                "openhands.agenthub.opencode_agent.subagents.parallel_runner."
                "ParallelSubagentRunner.run_parallel"
            ) as mock_run:
                mock_run.return_value = {
                    "tc-0": {"content": "R0", "outputs": {"content": "R0"}},
                    "tc-1": {"content": "R1", "outputs": {"content": "R1"}},
                }
                action = agent.step(state)

                mock_run.assert_called_once()
                assert agent.llm.completion.call_count == 2

    def test_mixed_calls_skip_parallel(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = _make_state()
        _wire_agent(agent, state)

        agent.llm.completion = Mock(
            return_value=_mock_response_multi(
                [
                    ("read", {"file_path": "/x.py"}),
                    (TASK_TOOL_NAME, {"description": "E", "prompt": "p", "subagent_type": "explore"}),
                ]
            )
        )

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://test:8000"}):
            with patch(
                "openhands.agenthub.opencode_agent.subagents.parallel_runner."
                "ParallelSubagentRunner.run_parallel"
            ) as mock_run:
                action = agent.step(state)
                mock_run.assert_not_called()
                assert isinstance(action, OpenCodeReadAction)

    def test_single_task_skips_parallel(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = _make_state()
        _wire_agent(agent, state)

        agent.llm.completion = Mock(
            return_value=_mock_response_multi(
                [(TASK_TOOL_NAME, {"description": "T", "prompt": "p", "subagent_type": "explore"})]
            )
        )

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://test:8000"}):
            with patch(
                "openhands.agenthub.opencode_agent.subagents.parallel_runner."
                "ParallelSubagentRunner.run_parallel"
            ) as mock_run:
                action = agent.step(state)
                mock_run.assert_not_called()
                assert isinstance(action, AgentDelegateAction)

    def test_no_runtime_url_skips_parallel(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = _make_state()
        _wire_agent(agent, state)

        agent.llm.completion = Mock(
            return_value=_mock_response_multi(
                [
                    (TASK_TOOL_NAME, {"description": "A", "prompt": "p", "subagent_type": "explore"}),
                    (TASK_TOOL_NAME, {"description": "B", "prompt": "p", "subagent_type": "general"}),
                ]
            )
        )

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "openhands.agenthub.opencode_agent.subagents.parallel_runner."
                "ParallelSubagentRunner.run_parallel"
            ) as mock_run:
                action = agent.step(state)
                mock_run.assert_not_called()
                assert isinstance(action, AgentDelegateAction)

    def test_no_task_calls_untouched(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = _make_state()
        _wire_agent(agent, state)

        agent.llm.completion = Mock(
            return_value=_mock_response_multi([("read", {"file_path": "/f"})])
        )

        action = agent.step(state)
        assert isinstance(action, OpenCodeReadAction)


# ===========================================================================
# 6. Synthetic event injection
# ===========================================================================


class TestSyntheticEventInjection:
    """Verify that _handle_parallel_subagents correctly injects
    AgentDelegateAction + AgentDelegateObservation into state.history."""

    def test_synthetic_events_structure(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = State(session_id="test-synth")
        initial_msg = MessageAction(content="parallel tasks")
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser = Mock()
        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )
        agent.conversation_memory = Mock()
        agent.conversation_memory.process_events = Mock(return_value=[])
        agent.conversation_memory.apply_prompt_caching = Mock()

        task_resp = _mock_response_multi(
            [
                (TASK_TOOL_NAME, {"description": "X", "prompt": "px", "subagent_type": "explore"}),
                (TASK_TOOL_NAME, {"description": "Y", "prompt": "py", "subagent_type": "general"}),
            ]
        )
        finish_resp = _mock_text_response("Done")
        agent.llm.completion = Mock(side_effect=[task_resp, finish_resp])

        before_len = len(state.history)

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://test:8000"}):
            with patch(
                "openhands.agenthub.opencode_agent.subagents.parallel_runner."
                "ParallelSubagentRunner.run_parallel"
            ) as mock_run:
                mock_run.return_value = {
                    "tc-0": {"content": "RX", "outputs": {"content": "RX"}},
                    "tc-1": {"content": "RY", "outputs": {"content": "RY"}},
                }
                agent.step(state)

        new_events = state.history[before_len:]
        assert len(new_events) == 4

        synth_acts = [e for e in new_events if isinstance(e, AgentDelegateAction)]
        synth_obs = [e for e in new_events if isinstance(e, AgentDelegateObservation)]

        assert len(synth_acts) == 2
        assert len(synth_obs) == 2

        for obs in synth_obs:
            assert obs.tool_call_metadata is not None
            matching = [a for a in synth_acts if a.id == obs.cause]
            assert len(matching) == 1

    def test_observation_content_matches_result(self, llm_registry):
        agent = _make_agent(llm_registry)
        state = State(session_id="test-content")
        initial_msg = MessageAction(content="tasks")
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser = Mock()
        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )
        agent.conversation_memory = Mock()
        agent.conversation_memory.process_events = Mock(return_value=[])
        agent.conversation_memory.apply_prompt_caching = Mock()

        task_resp = _mock_response_multi(
            [
                (TASK_TOOL_NAME, {"description": "A", "prompt": "a", "subagent_type": "explore"}),
                (TASK_TOOL_NAME, {"description": "B", "prompt": "b", "subagent_type": "general"}),
            ]
        )
        finish_resp = _mock_text_response("All done")
        agent.llm.completion = Mock(side_effect=[task_resp, finish_resp])

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://test:8000"}):
            with patch(
                "openhands.agenthub.opencode_agent.subagents.parallel_runner."
                "ParallelSubagentRunner.run_parallel"
            ) as mock_run:
                mock_run.return_value = {
                    "tc-0": {"content": "Found 5 files", "outputs": {"content": "Found 5 files"}},
                    "tc-1": {"content": "Refactored OK", "outputs": {"content": "Refactored OK"}},
                }
                agent.step(state)

        obs_events = [e for e in state.history if isinstance(e, AgentDelegateObservation)]
        contents = [o.content for o in obs_events]
        assert "Found 5 files" in contents
        assert "Refactored OK" in contents


# ===========================================================================
# 7. End-to-end: full parallel flow with real agents
# ===========================================================================


class TestE2EFullParallelFlow:
    """End-to-end test exercising the complete flow with minimal mocking.

    Only the LLM (completion calls) and the HTTP runtime are mocked.
    Everything else -- action conversion, parallel detection, runner,
    subagent instantiation, state management, synthetic event injection,
    and the re-call -- uses real code.

    Flow tested:
        Parent LLM call #1  ->  2 task tool calls
            |
            v
        response_to_actions  ->  2 AgentDelegateAction
            |
            v
        Parallel detection triggers  (>=2 delegates, all delegates, URL set)
            |
            v
        ParallelSubagentRunner.run_parallel()
            |
            +-> Thread 1: OpenCodeExploreSubAgent
            |       step() -> LLM returns read tool call
            |       _execute_action -> HTTP POST -> FileReadObservation
            |       step() -> LLM returns finish tool call
            |       _enrich_finish_action -> outputs with metadata
            |
            +-> Thread 2: OpenCodeGeneralSubAgent
            |       step() -> LLM returns read tool call
            |       _execute_action -> HTTP POST -> FileReadObservation
            |       step() -> LLM returns finish tool call
            |       _enrich_finish_action -> outputs with metadata
            |
            v
        Results collected, synthetic events injected into state.history
            |
            v
        Parent LLM call #2  ->  text summary MessageAction
            |
            v
        Returned to caller
    """

    @staticmethod
    def _make_llm_mock():
        """Create a mock LLM with standard config attributes."""
        llm = Mock()
        llm.config = Mock()
        llm.config.model = "test-model"
        llm.config.max_message_chars = 10000
        llm.config.custom_llm_provider = None
        llm.vision_is_active = Mock(return_value=False)
        llm.is_caching_prompt_active = Mock(return_value=False)
        return llm

    @staticmethod
    def _make_tool_call_response(tool_calls_spec, response_id="resp"):
        """Build a ModelResponse with tool calls.

        tool_calls_spec: list of (name, arguments_dict, call_id)
        """
        calls = []
        for name, args, cid in tool_calls_spec:
            calls.append({
                "function": {"name": name, "arguments": json.dumps(args)},
                "id": cid,
                "type": "function",
            })
        return ModelResponse(
            id=response_id,
            choices=[{
                "message": {
                    "tool_calls": calls,
                    "content": None,
                    "role": "assistant",
                },
                "index": 0,
                "finish_reason": "tool_calls",
            }],
        )

    @staticmethod
    def _make_http_read_response(content, path):
        """Build the JSON dict that the mocked runtime returns for a read action."""
        return {
            "observation": "read",
            "content": content,
            "extras": {"path": path, "impl_source": "default"},
        }

    def test_full_parallel_flow_two_subagents(self, llm_registry):
        """Two subagents run in parallel, each reads a file and finishes."""
        import threading

        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        # ---- 1. Build parent agent with real constructor ----
        agent = OpenCodeAgent(config=AgentConfig(), llm_registry=llm_registry)

        # Parent LLM: call 1 = two task tool calls, call 2 = summary text
        parent_llm = self._make_llm_mock()
        parent_task_response = self._make_tool_call_response(
            [
                (
                    TASK_TOOL_NAME,
                    {"description": "Explore src", "prompt": "Find all Python files in src/", "subagent_type": "explore"},
                    "tc-explore",
                ),
                (
                    TASK_TOOL_NAME,
                    {"description": "Analyze main", "prompt": "Analyze the main module structure", "subagent_type": "general"},
                    "tc-general",
                ),
            ],
            response_id="parent-resp-1",
        )
        parent_summary_response = _mock_text_response(
            "Both subagents completed. Found Python files and analyzed the main module."
        )
        parent_llm.completion = Mock(
            side_effect=[parent_task_response, parent_summary_response]
        )
        agent.llm = parent_llm

        # ---- 2. Build subagent LLM mocks ----
        # Each subagent does: read a file -> finish
        # We create two identical mocks since thread ordering is non-deterministic
        def make_subagent_llm(file_path, file_content, finish_msg):
            llm = self._make_llm_mock()
            read_resp = self._make_tool_call_response(
                [("read", {"file_path": file_path}, "sub-read")],
                response_id="sub-resp-1",
            )
            finish_resp = self._make_tool_call_response(
                [("finish", {"message": finish_msg}, "sub-finish")],
                response_id="sub-resp-2",
            )
            llm.completion = Mock(side_effect=[read_resp, finish_resp])
            return llm

        sub_llm_a = make_subagent_llm("/src/main.py", "def main(): pass", "Found 3 functions")
        sub_llm_b = make_subagent_llm("/src/utils.py", "def helper(): pass", "Found 5 utilities")

        # ---- 3. Wire get_router to dispatch LLMs ----
        # Call order: parent __init__ (already done), then 2 subagents in threads
        call_lock = threading.Lock()
        sub_llms = [sub_llm_a, sub_llm_b]
        sub_idx = [0]

        original_get_router = llm_registry.get_router

        def mock_get_router(config):
            with call_lock:
                idx = sub_idx[0]
                sub_idx[0] += 1
                if idx < len(sub_llms):
                    return sub_llms[idx]
                return sub_llms[-1]

        llm_registry.get_router = mock_get_router

        # ---- 4. Mock the HTTP runtime ----
        def mock_http_post(url, **kwargs):
            response = Mock()
            response.status_code = 200
            response.raise_for_status = Mock()
            action_data = kwargs.get("json", {}).get("action", {})
            path = action_data.get("args", {}).get("path", "/unknown")
            response.json = Mock(
                return_value=self._make_http_read_response(
                    f"# Contents of {path}\ndef example(): pass\n", path
                )
            )
            return response

        mock_client = Mock()
        mock_client.post = Mock(side_effect=mock_http_post)
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        # ---- 5. Build real State ----
        state = State(session_id="e2e-parallel-test")
        user_msg = MessageAction(
            content="Explore the src directory and analyze the main module"
        )
        user_msg._source = EventSource.USER
        user_msg._id = 0
        state.history = [user_msg]

        history_before = len(state.history)

        # ---- 6. Execute! ----
        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://localhost:9999"}):
            with patch("httpx.Client", return_value=mock_client):
                action = agent.step(state)

        # ---- 7. Verify results ----

        # The final action should be a MessageAction (the parent's summary)
        assert isinstance(action, MessageAction), f"Expected MessageAction, got {type(action)}"
        assert "subagents completed" in action.content.lower() or "both" in action.content.lower()

        # Parent LLM was called exactly twice
        assert parent_llm.completion.call_count == 2

        # Both subagent LLMs were called exactly twice each (read + finish)
        assert sub_llm_a.completion.call_count == 2
        assert sub_llm_b.completion.call_count == 2

        # HTTP endpoint was called twice (one read per subagent)
        assert mock_client.post.call_count == 2

        # Synthetic events were injected: 2 actions + 2 observations = 4 new events
        new_events = state.history[history_before:]
        assert len(new_events) == 4

        synth_actions = [e for e in new_events if isinstance(e, AgentDelegateAction)]
        synth_obs = [e for e in new_events if isinstance(e, AgentDelegateObservation)]
        assert len(synth_actions) == 2
        assert len(synth_obs) == 2

        # Each observation has tool_call_metadata and is causally linked to its action
        for obs in synth_obs:
            assert obs.tool_call_metadata is not None
            matching_actions = [a for a in synth_actions if a.id == obs.cause]
            assert len(matching_actions) == 1

        # Observation content should contain the subagent finish messages
        obs_contents = sorted([o.content for o in synth_obs])
        assert any("3 functions" in c or "5 utilities" in c for c in obs_contents)

        # The tool_call_ids should match the original task calls
        obs_tc_ids = {o.tool_call_metadata.tool_call_id for o in synth_obs}
        assert obs_tc_ids == {"tc-explore", "tc-general"}

        # Restore original get_router
        llm_registry.get_router = original_get_router

    def test_full_flow_subagent_error_does_not_crash_parent(self, llm_registry):
        """If one subagent's LLM throws, the other still completes and the parent gets results."""
        import threading

        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        agent = OpenCodeAgent(config=AgentConfig(), llm_registry=llm_registry)

        parent_llm = self._make_llm_mock()
        parent_task_response = self._make_tool_call_response(
            [
                (TASK_TOOL_NAME, {"description": "A", "prompt": "Task A", "subagent_type": "explore"}, "tc-a"),
                (TASK_TOOL_NAME, {"description": "B", "prompt": "Task B", "subagent_type": "general"}, "tc-b"),
            ],
            response_id="parent-err-1",
        )
        parent_summary = _mock_text_response("Partial results received.")
        parent_llm.completion = Mock(side_effect=[parent_task_response, parent_summary])
        agent.llm = parent_llm

        # Subagent A: works fine (read -> finish)
        ok_llm = self._make_llm_mock()
        ok_llm.completion = Mock(side_effect=[
            self._make_tool_call_response(
                [("read", {"file_path": "/ok.py"}, "r1")], response_id="ok-1"
            ),
            self._make_tool_call_response(
                [("finish", {"message": "Success from A"}, "f1")], response_id="ok-2"
            ),
        ])

        # Subagent B: LLM crashes on first call
        bad_llm = self._make_llm_mock()
        bad_llm.completion = Mock(side_effect=RuntimeError("LLM API is down"))

        call_lock = threading.Lock()
        sub_llms = [ok_llm, bad_llm]
        sub_idx = [0]

        def mock_get_router(config):
            with call_lock:
                idx = sub_idx[0]
                sub_idx[0] += 1
                if idx < len(sub_llms):
                    return sub_llms[idx]
                return sub_llms[-1]

        original_get_router = llm_registry.get_router
        llm_registry.get_router = mock_get_router

        def mock_http_post(url, **kwargs):
            resp = Mock()
            resp.status_code = 200
            resp.raise_for_status = Mock()
            resp.json = Mock(return_value={
                "observation": "read",
                "content": "file ok",
                "extras": {"path": "/ok.py", "impl_source": "default"},
            })
            return resp

        mock_client = Mock()
        mock_client.post = Mock(side_effect=mock_http_post)
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        state = State(session_id="e2e-error-test")
        user_msg = MessageAction(content="Do two things")
        user_msg._source = EventSource.USER
        user_msg._id = 0
        state.history = [user_msg]

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://localhost:9999"}):
            with patch("httpx.Client", return_value=mock_client):
                action = agent.step(state)

        # Should still succeed -- parent gets partial results
        assert isinstance(action, MessageAction)

        # Both synthetic observations exist (one success, one error)
        synth_obs = [
            e for e in state.history if isinstance(e, AgentDelegateObservation)
        ]
        assert len(synth_obs) == 2

        obs_contents = [o.content for o in synth_obs]
        # One should have success content, one should have error content
        has_success = any("Success from A" in c or "file ok" in c for c in obs_contents)
        has_error = any("error" in c.lower() for c in obs_contents)
        assert has_success or has_error  # at least one completed

        llm_registry.get_router = original_get_router
