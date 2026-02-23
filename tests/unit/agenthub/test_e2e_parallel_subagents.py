"""End-to-end test for the full parallel subagent flow.

Only the LLM and HTTP runtime are mocked. Everything else uses real code:
action conversion, parallel detection, ParallelSubagentRunner, real subagent
classes, SubagentMixin enrichment, state management, and synthetic events.
"""

import json
import threading
from unittest.mock import Mock, patch

import pytest
from litellm import ModelResponse

from openhands.agenthub.opencode_agent.tools.task import TASK_TOOL_NAME
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.events.action import AgentDelegateAction, MessageAction
from openhands.events.event import EventSource
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.llm.llm_registry import LLMRegistry


def _make_llm():
    llm = Mock()
    llm.config = Mock()
    llm.config.model = "test-model"
    llm.config.max_message_chars = 10000
    llm.config.custom_llm_provider = None
    llm.vision_is_active = Mock(return_value=False)
    llm.is_caching_prompt_active = Mock(return_value=False)
    return llm


def _tc_resp(specs, rid="r"):
    calls = [
        {
            "function": {"name": n, "arguments": json.dumps(a)},
            "id": cid,
            "type": "function",
        }
        for n, a, cid in specs
    ]
    return ModelResponse(
        id=rid,
        choices=[{
            "message": {"tool_calls": calls, "content": None, "role": "assistant"},
            "index": 0,
            "finish_reason": "tool_calls",
        }],
    )


def _txt_resp(text):
    return ModelResponse(
        id="txt",
        choices=[{
            "message": {"content": text, "role": "assistant", "tool_calls": None},
            "index": 0,
            "finish_reason": "stop",
        }],
    )


def _sub_llm(path, msg):
    llm = _make_llm()
    llm.completion = Mock(side_effect=[
        _tc_resp([("read", {"file_path": path}, "sr")], rid="s1"),
        _tc_resp([("finish", {"message": msg}, "sf")], rid="s2"),
    ])
    return llm


def _http_mock():
    def post(url, **kw):
        r = Mock()
        r.status_code = 200
        r.raise_for_status = Mock()
        p = kw.get("json", {}).get("action", {}).get("args", {}).get("path", "/x")
        r.json = Mock(return_value={
            "observation": "read",
            "content": f"contents of {p}",
            "extras": {"path": p, "impl_source": "default"},
        })
        return r

    c = Mock()
    c.post = Mock(side_effect=post)
    c.__enter__ = Mock(return_value=c)
    c.__exit__ = Mock(return_value=False)
    return c


def _router(llms):
    lock = threading.Lock()
    i = [0]

    def get(cfg):
        with lock:
            n = i[0]
            i[0] += 1
        return llms[min(n, len(llms) - 1)]

    return get


@pytest.fixture
def registry():
    cfg = OpenHandsConfig()
    cfg.set_llm_config(LLMConfig(model="gpt-4o", api_key="test"))
    return LLMRegistry(config=cfg)


class TestE2EParallelFlow:

    def test_two_subagents_happy_path(self, registry):
        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        agent = OpenCodeAgent(config=AgentConfig(), llm_registry=registry)
        p = _make_llm()
        p.completion = Mock(side_effect=[
            _tc_resp([
                (TASK_TOOL_NAME, {"description": "Explore", "prompt": "Find files", "subagent_type": "explore"}, "tc-e"),
                (TASK_TOOL_NAME, {"description": "Analyze", "prompt": "Analyze main", "subagent_type": "general"}, "tc-g"),
            ]),
            _txt_resp("Both subagents completed."),
        ])
        agent.llm = p

        sa = _sub_llm("/src/main.py", "Found 3 functions")
        sb = _sub_llm("/src/utils.py", "Found 5 helpers")
        orig = registry.get_router
        registry.get_router = _router([sa, sb])

        hc = _http_mock()
        state = State(session_id="e2e-1")
        um = MessageAction(content="Explore and analyze")
        um._source = EventSource.USER
        um._id = 0
        state.history = [um]
        before = len(state.history)

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://t:9"}):
            with patch("httpx.Client", return_value=hc):
                action = agent.step(state)

        assert isinstance(action, MessageAction)
        assert "completed" in action.content.lower()
        assert p.completion.call_count == 2
        assert sa.completion.call_count == 2
        assert sb.completion.call_count == 2
        assert hc.post.call_count == 2

        new = state.history[before:]
        assert len(new) == 4
        acts = [e for e in new if isinstance(e, AgentDelegateAction)]
        obs = [e for e in new if isinstance(e, AgentDelegateObservation)]
        assert len(acts) == 2
        assert len(obs) == 2

        for o in obs:
            assert o.tool_call_metadata is not None
            assert any(a.id == o.cause for a in acts)

        tc_ids = {o.tool_call_metadata.tool_call_id for o in obs}
        assert tc_ids == {"tc-e", "tc-g"}

        registry.get_router = orig

    def test_subagent_error_isolation(self, registry):
        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        agent = OpenCodeAgent(config=AgentConfig(), llm_registry=registry)
        p = _make_llm()
        p.completion = Mock(side_effect=[
            _tc_resp([
                (TASK_TOOL_NAME, {"description": "OK", "prompt": "A", "subagent_type": "explore"}, "tc-ok"),
                (TASK_TOOL_NAME, {"description": "Bad", "prompt": "B", "subagent_type": "general"}, "tc-bad"),
            ]),
            _txt_resp("Partial results."),
        ])
        agent.llm = p

        ok = _sub_llm("/ok.py", "Success")
        bad = _make_llm()
        bad.completion = Mock(side_effect=RuntimeError("crash"))
        orig = registry.get_router
        registry.get_router = _router([ok, bad])

        hc = _http_mock()
        state = State(session_id="e2e-err")
        um = MessageAction(content="Two tasks")
        um._source = EventSource.USER
        um._id = 0
        state.history = [um]

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://t:9"}):
            with patch("httpx.Client", return_value=hc):
                action = agent.step(state)

        assert isinstance(action, MessageAction)
        obs = [e for e in state.history if isinstance(e, AgentDelegateObservation)]
        assert len(obs) == 2
        contents = " ".join(o.content for o in obs).lower()
        assert "error" in contents or "success" in contents

        registry.get_router = orig

    def test_three_subagents(self, registry):
        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        agent = OpenCodeAgent(config=AgentConfig(), llm_registry=registry)
        p = _make_llm()
        p.completion = Mock(side_effect=[
            _tc_resp([
                (TASK_TOOL_NAME, {"description": "T1", "prompt": "P1", "subagent_type": "explore"}, "tc-1"),
                (TASK_TOOL_NAME, {"description": "T2", "prompt": "P2", "subagent_type": "explore"}, "tc-2"),
                (TASK_TOOL_NAME, {"description": "T3", "prompt": "P3", "subagent_type": "general"}, "tc-3"),
            ]),
            _txt_resp("All done."),
        ])
        agent.llm = p

        subs = [_sub_llm(f"/f{i}.py", f"Result {i}") for i in range(3)]
        orig = registry.get_router
        registry.get_router = _router(subs)

        hc = _http_mock()
        state = State(session_id="e2e-3")
        um = MessageAction(content="Three tasks")
        um._source = EventSource.USER
        um._id = 0
        state.history = [um]

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://t:9"}):
            with patch("httpx.Client", return_value=hc):
                action = agent.step(state)

        assert isinstance(action, MessageAction)
        assert p.completion.call_count == 2
        assert hc.post.call_count == 3

        obs = [e for e in state.history if isinstance(e, AgentDelegateObservation)]
        assert len(obs) == 3
        tc_ids = {o.tool_call_metadata.tool_call_id for o in obs}
        assert tc_ids == {"tc-1", "tc-2", "tc-3"}

        for s in subs:
            assert s.completion.call_count == 2

        registry.get_router = orig

    def test_enrichment_produces_task_metadata(self, registry):
        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        agent = OpenCodeAgent(config=AgentConfig(), llm_registry=registry)
        p = _make_llm()
        p.completion = Mock(side_effect=[
            _tc_resp([
                (TASK_TOOL_NAME, {"description": "Check", "prompt": "List configs", "subagent_type": "explore"}, "tc-m1"),
                (TASK_TOOL_NAME, {"description": "Update", "prompt": "Add hints", "subagent_type": "general"}, "tc-m2"),
            ]),
            _txt_resp("Done."),
        ])
        agent.llm = p

        sa = _sub_llm("/a.py", "Found configs")
        sb = _sub_llm("/b.py", "Added hints")
        orig = registry.get_router
        registry.get_router = _router([sa, sb])

        hc = _http_mock()
        state = State(session_id="e2e-meta")
        um = MessageAction(content="Check and update")
        um._source = EventSource.USER
        um._id = 0
        state.history = [um]

        with patch.dict("os.environ", {"OPENHANDS_RUNTIME_URL": "http://t:9"}):
            with patch("httpx.Client", return_value=hc):
                agent.step(state)

        obs = [e for e in state.history if isinstance(e, AgentDelegateObservation)]
        assert len(obs) == 2

        for o in obs:
            assert "<task_metadata>" in o.content
            assert "session_id:" in o.content
            out = o.outputs
            assert "output" in out
            assert "metadata" in out
            assert "title" in out
            assert "sessionId" in out["metadata"]

        registry.get_router = orig
