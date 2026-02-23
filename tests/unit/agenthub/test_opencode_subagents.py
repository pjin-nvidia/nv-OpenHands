"""Comprehensive end-to-end tests for the OpenCode subagent architecture.

Covers:
1. Task tool definition (schema validation)
2. Function calling: task tool -> AgentDelegateAction conversion
3. SubagentMixin: tool summary collection, title derivation, finish enrichment
4. Subagent classes: tool lists, step() enrichment, prompt managers
5. Agent registration and instantiation
6. E2E integration: parent delegates -> subagent runs -> outputs match opencode format
"""

import asyncio
import json
from collections import deque
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from litellm import ModelResponse

from openhands.agenthub.opencode_agent.function_calling import response_to_actions
from openhands.agenthub.opencode_agent.subagents.explore import (
    OpenCodeExploreSubAgent,
)
from openhands.agenthub.opencode_agent.subagents.general import (
    OpenCodeGeneralSubAgent,
)
from openhands.agenthub.opencode_agent.subagents.mixin import SubagentMixin
from openhands.agenthub.opencode_agent.tools.task import (
    SUBAGENT_NAME_MAP,
    SUBAGENT_REGISTRY,
    TASK_TOOL_NAME,
    TaskTool,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.schema import ActionType
from openhands.events.action import (
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    CmdRunAction,
    FileEditAction,
    GlobAction,
    GrepAction,
    ListDirAction,
    MessageAction,
    OpenCodeReadAction,
    OpenCodeWriteAction,
)
from openhands.events.action.agent import ValidationFailureAction
from openhands.events.event import EventSource, FileEditSource
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.tool import ToolCallMetadata
from openhands.llm.llm_registry import LLMRegistry


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_mock_response(function_name: str, arguments: dict) -> ModelResponse:
    """Create a mock LLM response with a single tool call."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': function_name,
                                'arguments': json.dumps(arguments),
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def create_mock_response_with_thought(
    function_name: str, arguments: dict, thought: str
) -> ModelResponse:
    """Create a mock LLM response with thought content and a tool call."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': function_name,
                                'arguments': json.dumps(arguments),
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': thought,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def create_mock_response_multi_tool(
    tool_calls: list[tuple[str, dict]],
) -> ModelResponse:
    """Create a mock LLM response with multiple tool calls."""
    calls = []
    for i, (name, args) in enumerate(tool_calls):
        calls.append(
            {
                'function': {
                    'name': name,
                    'arguments': json.dumps(args),
                },
                'id': f'mock-tool-call-id-{i}',
                'type': 'function',
            }
        )
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': calls,
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def _make_action_with_metadata(
    action_cls, function_name: str, tool_call_id: str = 'call_001', **kwargs
):
    """Create an action instance with tool_call_metadata attached (simulating what function_calling produces)."""
    action = action_cls(**kwargs)
    mock_response = ModelResponse(
        id='mock-resp',
        choices=[
            {
                'message': {'content': None, 'role': 'assistant', 'tool_calls': []},
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    action.tool_call_metadata = ToolCallMetadata(
        tool_call_id=tool_call_id,
        function_name=function_name,
        model_response=mock_response,
        total_calls_in_response=1,
    )
    return action


def _make_mock_state(
    history=None, inputs=None, session_id='test-session-123'
) -> Mock:
    """Create a mock State with the given history and inputs."""
    state = Mock(spec=State)
    state.history = history or []
    state.inputs = inputs or {}
    state.session_id = session_id
    state.extra_data = {}
    return state


@pytest.fixture
def create_llm_registry():
    def _get_registry(llm_config=None):
        if llm_config is None:
            llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = OpenHandsConfig()
        config.set_llm_config(llm_config)
        return LLMRegistry(config=config)

    return _get_registry


# ==============================================================================
# 1. Task Tool Definition Tests
# ==============================================================================


class TestTaskToolDefinition:
    """Tests for the task tool schema and configuration."""

    def test_task_tool_type(self):
        assert TaskTool['type'] == 'function'

    def test_task_tool_name(self):
        func = TaskTool['function']
        assert func['name'] == TASK_TOOL_NAME
        assert func['name'] == 'task'

    def test_task_tool_has_parameters(self):
        func = TaskTool['function']
        assert 'parameters' in func

    def test_task_tool_required_params(self):
        params = TaskTool['function']['parameters']
        assert set(params['required']) == {'description', 'prompt', 'subagent_type'}

    def test_task_tool_description_param(self):
        props = TaskTool['function']['parameters']['properties']
        assert 'description' in props
        assert props['description']['type'] == 'string'

    def test_task_tool_prompt_param(self):
        props = TaskTool['function']['parameters']['properties']
        assert 'prompt' in props
        assert props['prompt']['type'] == 'string'

    def test_task_tool_subagent_type_param(self):
        props = TaskTool['function']['parameters']['properties']
        assert 'subagent_type' in props
        assert props['subagent_type']['type'] == 'string'
        assert 'enum' in props['subagent_type']

    def test_task_tool_subagent_type_enum_values(self):
        props = TaskTool['function']['parameters']['properties']
        enum_vals = props['subagent_type']['enum']
        assert 'general' in enum_vals
        assert 'explore' in enum_vals
        assert len(enum_vals) == len(SUBAGENT_REGISTRY)

    def test_task_tool_description_contains_agents(self):
        desc = TaskTool['function']['description']
        assert 'general' in desc.lower()
        assert 'explore' in desc.lower()

    def test_task_tool_description_contains_usage_guidelines(self):
        desc = TaskTool['function']['description']
        assert 'When to use' in desc
        assert 'When NOT to use' in desc

    def test_subagent_registry_keys(self):
        assert 'general' in SUBAGENT_REGISTRY
        assert 'explore' in SUBAGENT_REGISTRY

    def test_subagent_name_map_matches_registry(self):
        assert set(SUBAGENT_NAME_MAP.keys()) == set(SUBAGENT_REGISTRY.keys())

    def test_subagent_name_map_values(self):
        assert SUBAGENT_NAME_MAP['general'] == 'OpenCodeGeneralSubAgent'
        assert SUBAGENT_NAME_MAP['explore'] == 'OpenCodeExploreSubAgent'


# ==============================================================================
# 2. Function Calling: Task Tool -> AgentDelegateAction
# ==============================================================================


class TestTaskToolFunctionCalling:
    """Tests for converting task tool calls to AgentDelegateAction."""

    def test_task_tool_general_creates_delegate_action(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Search codebase',
                'prompt': 'Find all uses of the Config class',
                'subagent_type': 'general',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, AgentDelegateAction)
        assert action.agent == 'OpenCodeGeneralSubAgent'
        assert action.inputs['task'] == 'Find all uses of the Config class'
        assert action.inputs['description'] == 'Search codebase'

    def test_task_tool_explore_creates_delegate_action(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Explore project',
                'prompt': 'Find all Python files in src/',
                'subagent_type': 'explore',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, AgentDelegateAction)
        assert action.agent == 'OpenCodeExploreSubAgent'
        assert action.inputs['task'] == 'Find all Python files in src/'
        assert action.inputs['description'] == 'Explore project'

    def test_task_tool_missing_prompt_raises_validation_error(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Test',
                'subagent_type': 'general',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)
        assert 'prompt' in actions[0].error_message.lower()

    def test_task_tool_missing_subagent_type_raises_validation_error(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Test',
                'prompt': 'Do something',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)
        assert 'subagent_type' in actions[0].error_message.lower()

    def test_task_tool_invalid_subagent_type_raises_validation_error(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Test',
                'prompt': 'Do something',
                'subagent_type': 'nonexistent_agent',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], ValidationFailureAction)
        assert 'nonexistent_agent' in actions[0].error_message

    def test_task_tool_with_thought(self):
        response = create_mock_response_with_thought(
            TASK_TOOL_NAME,
            {
                'description': 'Explore structure',
                'prompt': 'Map the directory layout',
                'subagent_type': 'explore',
            },
            thought='Let me delegate this to the explore agent.',
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, AgentDelegateAction)
        assert action.thought == 'Let me delegate this to the explore agent.'

    def test_task_tool_description_defaults_to_empty(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'prompt': 'Find all tests',
                'subagent_type': 'explore',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, AgentDelegateAction)
        assert action.inputs['description'] == ''

    def test_task_tool_has_tool_call_metadata(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Test task',
                'prompt': 'Do something',
                'subagent_type': 'general',
            },
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert action.tool_call_metadata is not None
        assert action.tool_call_metadata.function_name == TASK_TOOL_NAME
        assert action.tool_call_metadata.tool_call_id == 'mock-tool-call-id'

    def test_task_tool_action_type_is_delegate(self):
        response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Test',
                'prompt': 'Do it',
                'subagent_type': 'general',
            },
        )
        actions = response_to_actions(response)
        assert actions[0].action == ActionType.DELEGATE

    def test_task_tool_among_multiple_tool_calls(self):
        """Test task tool call when it appears alongside other tool calls."""
        response = create_mock_response_multi_tool(
            [
                (
                    TASK_TOOL_NAME,
                    {
                        'description': 'Explore code',
                        'prompt': 'Find all classes',
                        'subagent_type': 'explore',
                    },
                ),
                (
                    TASK_TOOL_NAME,
                    {
                        'description': 'Implement feature',
                        'prompt': 'Add logging to main.py',
                        'subagent_type': 'general',
                    },
                ),
            ]
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert isinstance(actions[0], AgentDelegateAction)
        assert actions[0].agent == 'OpenCodeExploreSubAgent'
        assert isinstance(actions[1], AgentDelegateAction)
        assert actions[1].agent == 'OpenCodeGeneralSubAgent'


# ==============================================================================
# 3. SubagentMixin Tests
# ==============================================================================


class TestSubagentMixin:
    """Tests for SubagentMixin: tool summary collection and finish action enrichment."""

    def _make_mixin_instance(self, model_name='gpt-4o'):
        """Create a SubagentMixin instance with a mock LLM."""
        mixin = SubagentMixin()
        mixin.llm = Mock()
        mixin.llm.config = Mock()
        mixin.llm.config.model = model_name
        return mixin

    # -- _derive_tool_title --

    def test_derive_title_read(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.path = '/src/main.py'
        assert mixin._derive_tool_title(action, 'read') == 'Read /src/main.py'

    def test_derive_title_write(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.path = '/src/new.py'
        assert mixin._derive_tool_title(action, 'write') == 'Write /src/new.py'

    def test_derive_title_edit(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.path = '/src/config.py'
        assert mixin._derive_tool_title(action, 'edit') == 'Edit /src/config.py'

    def test_derive_title_glob(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.pattern = '**/*.py'
        assert mixin._derive_tool_title(action, 'glob') == 'Glob **/*.py'

    def test_derive_title_grep(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.pattern = 'import os'
        assert mixin._derive_tool_title(action, 'grep') == 'Grep import os'

    def test_derive_title_list_dir(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.path = '/src'
        assert mixin._derive_tool_title(action, 'list_dir') == 'List /src'

    def test_derive_title_bash(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.command = 'git log --oneline -10'
        title = mixin._derive_tool_title(action, 'execute_bash')
        assert title == 'Bash: git log --oneline -10'

    def test_derive_title_bash_truncates(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.command = 'x' * 100
        title = mixin._derive_tool_title(action, 'cmd_run')
        assert len(title) <= 66  # "Bash: " + 60 chars

    def test_derive_title_think(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.thought = 'Let me reason about the bug in the parser module carefully'
        title = mixin._derive_tool_title(action, 'think')
        assert title.startswith('Think: ')
        assert len(title) <= 48  # "Think: " + 40 chars

    def test_derive_title_unknown_tool(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        assert mixin._derive_tool_title(action, 'unknown_tool') == ''

    def test_derive_title_missing_attribute(self):
        mixin = self._make_mixin_instance()
        action = Mock(spec=[])  # No attributes
        assert mixin._derive_tool_title(action, 'read') == ''

    # -- _collect_tool_summary --

    def test_collect_summary_empty_history(self):
        mixin = self._make_mixin_instance()
        state = _make_mock_state(history=[])
        summary = mixin._collect_tool_summary(state)
        assert summary == []

    def test_collect_summary_skips_events_without_metadata(self):
        mixin = self._make_mixin_instance()
        event = Mock()
        del event.tool_call_metadata  # hasattr will return False
        state = _make_mock_state(history=[event])
        summary = mixin._collect_tool_summary(state)
        assert summary == []

    def test_collect_summary_skips_none_metadata(self):
        mixin = self._make_mixin_instance()
        event = Mock()
        event.tool_call_metadata = None
        state = _make_mock_state(history=[event])
        summary = mixin._collect_tool_summary(state)
        assert summary == []

    def test_collect_summary_skips_observations(self):
        """Observations have tool_call_metadata but should be skipped."""
        mixin = self._make_mixin_instance()
        from openhands.events.observation.observation import Observation

        obs = Mock(spec=Observation)
        obs.tool_call_metadata = ToolCallMetadata(
            tool_call_id='call_1',
            function_name='read',
            model_response=ModelResponse(id='r', choices=[]),
            total_calls_in_response=1,
        )
        state = _make_mock_state(history=[obs])
        summary = mixin._collect_tool_summary(state)
        assert summary == []

    def test_collect_summary_single_action(self):
        mixin = self._make_mixin_instance()
        action = _make_action_with_metadata(
            OpenCodeReadAction,
            'read',
            tool_call_id='call_001',
            path='/test.py',
            offset=0,
            limit=2000,
        )
        state = _make_mock_state(history=[action])
        summary = mixin._collect_tool_summary(state)
        assert len(summary) == 1
        assert summary[0]['id'] == 'call_001'
        assert summary[0]['tool'] == 'read'
        assert summary[0]['state']['status'] == 'completed'
        assert summary[0]['state']['title'] == 'Read /test.py'

    def test_collect_summary_multiple_actions(self):
        mixin = self._make_mixin_instance()
        actions = [
            _make_action_with_metadata(
                OpenCodeReadAction,
                'read',
                tool_call_id='call_001',
                path='/a.py',
                offset=0,
                limit=2000,
            ),
            _make_action_with_metadata(
                GrepAction,
                'grep',
                tool_call_id='call_002',
                pattern='import',
                path='.',
                include='',
            ),
            _make_action_with_metadata(
                GlobAction,
                'glob',
                tool_call_id='call_003',
                pattern='**/*.py',
                path='.',
            ),
        ]
        state = _make_mock_state(history=actions)
        summary = mixin._collect_tool_summary(state)
        assert len(summary) == 3
        assert summary[0]['tool'] == 'read'
        assert summary[1]['tool'] == 'grep'
        assert summary[2]['tool'] == 'glob'

    def test_collect_summary_skips_none_function_name(self):
        mixin = self._make_mixin_instance()
        action = Mock()
        action.tool_call_metadata = ToolCallMetadata(
            tool_call_id='call_x',
            function_name=None,
            model_response=ModelResponse(id='r', choices=[]),
            total_calls_in_response=1,
        )
        state = _make_mock_state(history=[action])
        summary = mixin._collect_tool_summary(state)
        assert summary == []

    # -- _enrich_finish_action --

    def test_enrich_finish_basic(self):
        mixin = self._make_mixin_instance(model_name='claude-3-opus')
        finish = AgentFinishAction(final_thought='Found 3 files matching the pattern.')
        state = _make_mock_state(
            history=[],
            inputs={'description': 'Search codebase', 'task': 'Find files'},
            session_id='session-abc',
        )
        result = mixin._enrich_finish_action(finish, state)

        assert result.outputs['title'] == 'Search codebase'
        assert result.outputs['metadata']['sessionId'] == 'session-abc'
        assert result.outputs['metadata']['model'] == 'claude-3-opus'
        assert result.outputs['metadata']['summary'] == []
        assert '<task_metadata>' in result.outputs['output']
        assert 'session_id: session-abc' in result.outputs['output']
        assert '</task_metadata>' in result.outputs['output']
        assert 'Found 3 files matching the pattern.' in result.outputs['output']

    def test_enrich_finish_content_key_for_conversation_memory(self):
        """The 'content' key is used by ConversationMemory for AgentDelegateObservation."""
        mixin = self._make_mixin_instance()
        finish = AgentFinishAction(final_thought='Done.')
        state = _make_mock_state(
            inputs={'description': 'Test'},
            session_id='sess-1',
        )
        result = mixin._enrich_finish_action(finish, state)

        assert 'content' in result.outputs
        assert result.outputs['content'] == result.outputs['output']
        assert 'Done.' in result.outputs['content']
        assert '<task_metadata>' in result.outputs['content']

    def test_enrich_finish_with_tool_history(self):
        mixin = self._make_mixin_instance()
        read_action = _make_action_with_metadata(
            OpenCodeReadAction,
            'read',
            tool_call_id='call_01',
            path='/config.py',
            offset=0,
            limit=2000,
        )
        grep_action = _make_action_with_metadata(
            GrepAction,
            'grep',
            tool_call_id='call_02',
            pattern='API_KEY',
            path='.',
            include='',
        )
        finish = AgentFinishAction(final_thought='API_KEY found in config.py line 42.')
        state = _make_mock_state(
            history=[read_action, grep_action],
            inputs={'description': 'Find API key'},
            session_id='sess-2',
        )
        result = mixin._enrich_finish_action(finish, state)

        summary = result.outputs['metadata']['summary']
        assert len(summary) == 2
        assert summary[0]['tool'] == 'read'
        assert summary[0]['state']['title'] == 'Read /config.py'
        assert summary[1]['tool'] == 'grep'
        assert summary[1]['state']['title'] == 'Grep API_KEY'

    def test_enrich_finish_fallback_to_thought(self):
        """When final_thought is empty, fall back to thought field."""
        mixin = self._make_mixin_instance()
        finish = AgentFinishAction(thought='Completed via thought field.')
        state = _make_mock_state(inputs={}, session_id='s')
        result = mixin._enrich_finish_action(finish, state)
        assert 'Completed via thought field.' in result.outputs['output']

    def test_enrich_finish_empty_inputs(self):
        mixin = self._make_mixin_instance()
        finish = AgentFinishAction(final_thought='Done.')
        state = _make_mock_state(inputs={}, session_id='s')
        result = mixin._enrich_finish_action(finish, state)
        assert result.outputs['title'] == ''

    def test_enrich_finish_no_llm(self):
        """Mixin without llm attribute should still work (model defaults to empty)."""
        mixin = SubagentMixin()
        finish = AgentFinishAction(final_thought='Result.')
        state = _make_mock_state(inputs={}, session_id='s')
        result = mixin._enrich_finish_action(finish, state)
        assert result.outputs['metadata']['model'] == ''

    def test_output_format_matches_opencode(self):
        """Verify the output structure matches the opencode task tool return format."""
        mixin = self._make_mixin_instance(model_name='gpt-4o')
        action = _make_action_with_metadata(
            OpenCodeReadAction,
            'read',
            tool_call_id='call_abc',
            path='/main.py',
            offset=0,
            limit=2000,
        )
        finish = AgentFinishAction(final_thought='Analysis complete.')
        state = _make_mock_state(
            history=[action],
            inputs={'description': 'Analyze main', 'task': 'Read main.py'},
            session_id='session-xyz',
        )
        result = mixin._enrich_finish_action(finish, state)
        outputs = result.outputs

        # Verify all required keys exist
        assert 'title' in outputs
        assert 'metadata' in outputs
        assert 'output' in outputs
        assert 'content' in outputs

        # Verify metadata structure
        metadata = outputs['metadata']
        assert 'summary' in metadata
        assert 'sessionId' in metadata
        assert 'model' in metadata

        # Verify summary entry structure
        assert len(metadata['summary']) == 1
        entry = metadata['summary'][0]
        assert 'id' in entry
        assert 'tool' in entry
        assert 'state' in entry
        assert 'status' in entry['state']

        # Verify output text format
        output = outputs['output']
        assert output.startswith('Analysis complete.')
        assert '<task_metadata>\nsession_id: session-xyz\n</task_metadata>' in output


# ==============================================================================
# 4. Subagent Classes: Tool Lists and Step Enrichment
# ==============================================================================


class TestOpenCodeGeneralSubAgent:
    """Tests for OpenCodeGeneralSubAgent tool list and behavior."""

    @pytest.fixture
    def general_agent(self, create_llm_registry):
        llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = AgentConfig()
        agent = OpenCodeGeneralSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )
        agent.llm = Mock()
        agent.llm.config = Mock()
        agent.llm.config.model = 'gpt-4o'
        agent.llm.config.max_message_chars = 10000
        return agent

    def test_general_tools_include_file_ops(self, general_agent):
        tool_names = [t['function']['name'] for t in general_agent.tools]
        assert 'read' in tool_names
        assert 'write' in tool_names
        assert 'edit' in tool_names

    def test_general_tools_include_search(self, general_agent):
        tool_names = [t['function']['name'] for t in general_agent.tools]
        assert 'glob' in tool_names
        assert 'grep' in tool_names
        assert 'list_dir' in tool_names

    def test_general_tools_include_think(self, general_agent):
        tool_names = [t['function']['name'] for t in general_agent.tools]
        assert 'think' in tool_names

    def test_general_tools_include_finish(self, general_agent):
        tool_names = [t['function']['name'] for t in general_agent.tools]
        assert 'finish' in tool_names

    def test_general_tools_exclude_todo(self, general_agent):
        tool_names = [t['function']['name'] for t in general_agent.tools]
        assert 'todo_read' not in tool_names
        assert 'todo_write' not in tool_names

    def test_general_tools_exclude_task(self, general_agent):
        tool_names = [t['function']['name'] for t in general_agent.tools]
        assert 'task' not in tool_names

    def test_general_prompt_manager(self, general_agent):
        pm = general_agent.prompt_manager
        assert pm is not None

    def test_general_is_subagent_mixin(self, general_agent):
        assert isinstance(general_agent, SubagentMixin)

    def test_general_is_opencode_agent(self, general_agent):
        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        assert isinstance(general_agent, OpenCodeAgent)


class TestOpenCodeExploreSubAgent:
    """Tests for OpenCodeExploreSubAgent tool list and behavior."""

    @pytest.fixture
    def explore_agent(self, create_llm_registry):
        llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = AgentConfig()
        agent = OpenCodeExploreSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )
        agent.llm = Mock()
        agent.llm.config = Mock()
        agent.llm.config.model = 'gpt-4o'
        agent.llm.config.max_message_chars = 10000
        return agent

    def test_explore_tools_include_read_only(self, explore_agent):
        tool_names = [t['function']['name'] for t in explore_agent.tools]
        assert 'read' in tool_names
        assert 'glob' in tool_names
        assert 'grep' in tool_names
        assert 'list_dir' in tool_names

    def test_explore_tools_include_think(self, explore_agent):
        tool_names = [t['function']['name'] for t in explore_agent.tools]
        assert 'think' in tool_names

    def test_explore_tools_include_finish(self, explore_agent):
        tool_names = [t['function']['name'] for t in explore_agent.tools]
        assert 'finish' in tool_names

    def test_explore_tools_exclude_write_ops(self, explore_agent):
        tool_names = [t['function']['name'] for t in explore_agent.tools]
        assert 'write' not in tool_names
        assert 'edit' not in tool_names

    def test_explore_tools_exclude_todo(self, explore_agent):
        tool_names = [t['function']['name'] for t in explore_agent.tools]
        assert 'todo_read' not in tool_names
        assert 'todo_write' not in tool_names

    def test_explore_tools_exclude_task(self, explore_agent):
        tool_names = [t['function']['name'] for t in explore_agent.tools]
        assert 'task' not in tool_names

    def test_explore_prompt_manager(self, explore_agent):
        pm = explore_agent.prompt_manager
        assert pm is not None

    def test_explore_is_subagent_mixin(self, explore_agent):
        assert isinstance(explore_agent, SubagentMixin)


# ==============================================================================
# 5. Agent Registration Tests
# ==============================================================================


class TestAgentRegistration:
    """Tests that subagent classes are properly registered."""

    def test_general_subagent_registered(self):
        import openhands.agenthub.opencode_agent  # noqa: F401 - triggers registration

        cls = Agent.get_cls('OpenCodeGeneralSubAgent')
        assert cls is not None

    def test_explore_subagent_registered(self):
        import openhands.agenthub.opencode_agent  # noqa: F401 - triggers registration

        cls = Agent.get_cls('OpenCodeExploreSubAgent')
        assert cls is not None

    def test_primary_agent_still_registered(self):
        import openhands.agenthub.opencode_agent  # noqa: F401 - triggers registration

        cls = Agent.get_cls('OpenCodeAgent')
        assert cls is not None

    def test_general_subagent_class_identity(self):
        import openhands.agenthub.opencode_agent  # noqa: F401

        cls = Agent.get_cls('OpenCodeGeneralSubAgent')
        assert cls is OpenCodeGeneralSubAgent or (
            callable(cls) and 'General' in str(cls)
        )

    def test_explore_subagent_class_identity(self):
        import openhands.agenthub.opencode_agent  # noqa: F401

        cls = Agent.get_cls('OpenCodeExploreSubAgent')
        assert cls is OpenCodeExploreSubAgent or (
            callable(cls) and 'Explore' in str(cls)
        )

    def test_general_subagent_can_instantiate(self, create_llm_registry):
        llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = AgentConfig()
        agent = OpenCodeGeneralSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )
        assert agent is not None
        assert agent.tools is not None

    def test_explore_subagent_can_instantiate(self, create_llm_registry):
        llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = AgentConfig()
        agent = OpenCodeExploreSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )
        assert agent is not None
        assert agent.tools is not None


# ==============================================================================
# 6. E2E Integration: Full Delegation Flow with Output Format Verification
# ==============================================================================


class TestE2ESubagentDelegation:
    """End-to-end tests for the subagent delegation flow.

    Verifies: parent calls task tool -> AgentDelegateAction created ->
    subagent runs -> AgentFinishAction enriched -> outputs match opencode format.
    """

    def test_full_flow_explore_subagent(self, create_llm_registry):
        """Simulate a full explore subagent flow: task tool call -> delegate -> finish with enriched outputs."""
        # Step 1: Parent LLM returns a task tool call
        parent_response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Explore test structure',
                'prompt': 'Find all test files and describe the test organization',
                'subagent_type': 'explore',
            },
        )
        parent_actions = response_to_actions(parent_response)
        assert len(parent_actions) == 1
        delegate_action = parent_actions[0]
        assert isinstance(delegate_action, AgentDelegateAction)
        assert delegate_action.agent == 'OpenCodeExploreSubAgent'

        # Step 2: The controller would create a child State with the inputs
        child_state = _make_mock_state(
            inputs=delegate_action.inputs,
            session_id='delegate-session-001',
        )

        # Step 3: Simulate the explore subagent executing some tools
        read_action = _make_action_with_metadata(
            OpenCodeReadAction,
            'read',
            tool_call_id='call_e1',
            path='/tests/conftest.py',
            offset=0,
            limit=2000,
        )
        glob_action = _make_action_with_metadata(
            GlobAction,
            'glob',
            tool_call_id='call_e2',
            pattern='tests/**/*.py',
            path='.',
        )
        grep_action = _make_action_with_metadata(
            GrepAction,
            'grep',
            tool_call_id='call_e3',
            pattern='def test_',
            path='.',
            include='',
        )
        child_state.history = [read_action, glob_action, grep_action]

        # Step 4: The explore subagent calls finish
        finish_action = AgentFinishAction(
            final_thought='Found 45 test files organized in tests/unit/ and tests/e2e/. '
            'Key test categories: agenthub, controller, memory, runtime.'
        )

        # Step 5: SubagentMixin enriches the finish action
        llm_config = LLMConfig(model='claude-3-opus', api_key='test_key')
        config = AgentConfig()
        explore_agent = OpenCodeExploreSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )
        explore_agent.llm = Mock()
        explore_agent.llm.config = Mock()
        explore_agent.llm.config.model = 'claude-3-opus'

        enriched = explore_agent._enrich_finish_action(finish_action, child_state)

        # Step 6: Verify the outputs match the opencode format exactly
        outputs = enriched.outputs

        # Title from inputs
        assert outputs['title'] == 'Explore test structure'

        # Metadata
        metadata = outputs['metadata']
        assert metadata['sessionId'] == 'delegate-session-001'
        assert metadata['model'] == 'claude-3-opus'

        # Summary of tool executions
        summary = metadata['summary']
        assert len(summary) == 3
        assert summary[0] == {
            'id': 'call_e1',
            'tool': 'read',
            'state': {'status': 'completed', 'title': 'Read /tests/conftest.py'},
        }
        assert summary[1] == {
            'id': 'call_e2',
            'tool': 'glob',
            'state': {'status': 'completed', 'title': 'Glob tests/**/*.py'},
        }
        assert summary[2] == {
            'id': 'call_e3',
            'tool': 'grep',
            'state': {'status': 'completed', 'title': 'Grep def test_'},
        }

        # Output text with task_metadata block
        output = outputs['output']
        assert output.startswith('Found 45 test files')
        assert '<task_metadata>\nsession_id: delegate-session-001\n</task_metadata>' in output

        # Content key (used by ConversationMemory)
        assert outputs['content'] == outputs['output']

    def test_full_flow_general_subagent(self, create_llm_registry):
        """Simulate a full general subagent flow with write operations."""
        # Step 1: Task tool call
        parent_response = create_mock_response(
            TASK_TOOL_NAME,
            {
                'description': 'Add logging',
                'prompt': 'Add debug logging to the main module',
                'subagent_type': 'general',
            },
        )
        parent_actions = response_to_actions(parent_response)
        delegate_action = parent_actions[0]
        assert delegate_action.agent == 'OpenCodeGeneralSubAgent'

        # Step 2: Child state
        child_state = _make_mock_state(
            inputs=delegate_action.inputs,
            session_id='delegate-session-002',
        )

        # Step 3: General subagent executes read, then edit
        read_action = _make_action_with_metadata(
            OpenCodeReadAction,
            'read',
            tool_call_id='call_g1',
            path='/src/main.py',
            offset=0,
            limit=2000,
        )
        edit_action = _make_action_with_metadata(
            FileEditAction,
            'edit',
            tool_call_id='call_g2',
            path='/src/main.py',
            command='str_replace',
            old_str='def main():',
            new_str='def main():\n    logger.debug("Starting main")',
            impl_source=FileEditSource.OH_ACI,
        )
        child_state.history = [read_action, edit_action]

        # Step 4: Finish
        finish_action = AgentFinishAction(
            final_thought='Added debug logging to main() function in /src/main.py.'
        )

        # Step 5: Enrich
        llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = AgentConfig()
        general_agent = OpenCodeGeneralSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )
        general_agent.llm = Mock()
        general_agent.llm.config = Mock()
        general_agent.llm.config.model = 'gpt-4o'

        enriched = general_agent._enrich_finish_action(finish_action, child_state)
        outputs = enriched.outputs

        assert outputs['title'] == 'Add logging'
        assert len(outputs['metadata']['summary']) == 2
        assert outputs['metadata']['summary'][0]['tool'] == 'read'
        assert outputs['metadata']['summary'][1]['tool'] == 'edit'
        assert outputs['metadata']['summary'][1]['state']['title'] == 'Edit /src/main.py'
        assert 'Added debug logging' in outputs['output']
        assert '<task_metadata>' in outputs['output']

    def test_delegate_observation_content_uses_output(self):
        """Verify that AgentDelegateObservation.outputs['content'] contains the
        opencode-formatted text, which ConversationMemory will use."""
        outputs = {
            'content': 'Analysis done.\n\n<task_metadata>\nsession_id: s1\n</task_metadata>',
            'title': 'Test',
            'metadata': {
                'summary': [{'id': 'c1', 'tool': 'read', 'state': {'status': 'completed'}}],
                'sessionId': 's1',
                'model': 'gpt-4o',
            },
            'output': 'Analysis done.\n\n<task_metadata>\nsession_id: s1\n</task_metadata>',
        }
        obs = AgentDelegateObservation(
            outputs=outputs,
            content='Delegated agent finished with result:\n\nTest agent done',
        )

        # ConversationMemory reads obs.outputs.get('content', obs.content)
        effective_content = obs.outputs.get('content', obs.content)
        assert effective_content == outputs['content']
        assert '<task_metadata>' in effective_content
        assert 'session_id: s1' in effective_content

    def test_parallel_task_tool_calls(self):
        """Test that multiple concurrent task tool calls produce independent delegate actions."""
        response = create_mock_response_multi_tool(
            [
                (
                    TASK_TOOL_NAME,
                    {
                        'description': 'Explore backend',
                        'prompt': 'Find all API routes in the backend',
                        'subagent_type': 'explore',
                    },
                ),
                (
                    TASK_TOOL_NAME,
                    {
                        'description': 'Explore frontend',
                        'prompt': 'Find all React components',
                        'subagent_type': 'explore',
                    },
                ),
            ]
        )
        actions = response_to_actions(response)
        assert len(actions) == 2

        assert isinstance(actions[0], AgentDelegateAction)
        assert actions[0].agent == 'OpenCodeExploreSubAgent'
        assert actions[0].inputs['description'] == 'Explore backend'

        assert isinstance(actions[1], AgentDelegateAction)
        assert actions[1].agent == 'OpenCodeExploreSubAgent'
        assert actions[1].inputs['description'] == 'Explore frontend'

        # Each has its own tool_call_id
        assert actions[0].tool_call_metadata.tool_call_id != actions[1].tool_call_metadata.tool_call_id

    def test_mixed_task_and_regular_tool_calls(self):
        """Test task tool mixed with regular tools in a single response."""
        response = create_mock_response_multi_tool(
            [
                (
                    'read',
                    {'file_path': '/README.md'},
                ),
                (
                    TASK_TOOL_NAME,
                    {
                        'description': 'Deep analysis',
                        'prompt': 'Analyze the codebase architecture',
                        'subagent_type': 'general',
                    },
                ),
            ]
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert isinstance(actions[0], OpenCodeReadAction)
        assert isinstance(actions[1], AgentDelegateAction)
        assert actions[1].agent == 'OpenCodeGeneralSubAgent'

    def test_subagent_step_enriches_finish_action(self, create_llm_registry):
        """Test that calling step() on a subagent with a pending AgentFinishAction
        enriches the outputs correctly."""
        llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
        config = AgentConfig()
        agent = OpenCodeExploreSubAgent(
            config=config, llm_registry=create_llm_registry(llm_config)
        )

        # Mock the LLM to return a finish tool call
        mock_llm = Mock()
        mock_llm.config = Mock()
        mock_llm.config.model = 'gpt-4o'
        mock_llm.config.max_message_chars = 10000
        mock_llm.vision_is_active = Mock(return_value=False)
        mock_llm.is_caching_prompt_active = Mock(return_value=False)
        mock_llm.completion = Mock(
            return_value=create_mock_response('finish', {'message': 'Done exploring.'})
        )
        agent.llm = mock_llm

        # Create a state with history containing a read action
        state = Mock(spec=State)
        state.history = [
            _make_action_with_metadata(
                OpenCodeReadAction,
                'read',
                tool_call_id='call_r1',
                path='/src/app.py',
                offset=0,
                limit=2000,
            ),
        ]
        state.inputs = {'description': 'Explore app', 'task': 'Read app.py'}
        state.session_id = 'sess-step-test'
        state.extra_data = {}

        # Simulate get_last_user_message
        user_msg = Mock()
        user_msg.content = 'TASK: Read app.py'
        state.get_last_user_message = Mock(return_value=user_msg)

        # Mock condenser to return the history as-is
        from openhands.memory.condenser.condenser import View

        agent.condenser = Mock()
        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )

        # Mock _get_initial_user_message
        initial_msg = MessageAction(content='TASK: Read app.py')
        initial_msg._source = EventSource.USER
        state.history.insert(0, initial_msg)

        # Mock conversation_memory
        agent.conversation_memory = Mock()
        agent.conversation_memory.process_events = Mock(return_value=[])
        agent.conversation_memory.apply_prompt_caching = Mock()

        # Step should call LLM and get finish action, which gets enriched
        action = agent.step(state)

        assert isinstance(action, AgentFinishAction)
        assert 'content' in action.outputs
        assert 'metadata' in action.outputs
        assert 'output' in action.outputs
        assert '<task_metadata>' in action.outputs['output']
        assert 'session_id: sess-step-test' in action.outputs['output']
        # The summary should include the read action from history
        summary = action.outputs['metadata']['summary']
        assert len(summary) == 1
        assert summary[0]['tool'] == 'read'
        assert summary[0]['state']['title'] == 'Read /src/app.py'


# ==============================================================================
# Parallel Subagent Runner Tests
# ==============================================================================


class TestParallelSubagentRunner:
    """Tests for the ParallelSubagentRunner."""

    def _make_delegate_action(
        self, agent_name: str, task: str, tool_call_id: str
    ) -> AgentDelegateAction:
        """Create an AgentDelegateAction with tool_call_metadata."""
        action = AgentDelegateAction(
            agent=agent_name,
            inputs={'task': task, 'description': f'Test: {task}'},
        )
        action.tool_call_metadata = ToolCallMetadata(
            tool_call_id=tool_call_id,
            function_name='task',
            model_response=ModelResponse(
                id='mock-resp',
                choices=[
                    {
                        'message': {
                            'tool_calls': [
                                {
                                    'function': {
                                        'name': 'task',
                                        'arguments': json.dumps(
                                            {
                                                'prompt': task,
                                                'subagent_type': 'general',
                                            }
                                        ),
                                    },
                                    'id': tool_call_id,
                                    'type': 'function',
                                }
                            ],
                            'content': None,
                            'role': 'assistant',
                        },
                        'index': 0,
                        'finish_reason': 'tool_calls',
                    }
                ],
            ),
            total_calls_in_response=2,
        )
        return action

    def test_runner_requires_runtime_url(self):
        """Runner raises when OPENHANDS_RUNTIME_URL is not set."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url=None,
        )
        # Clear env in case it's set
        with patch.dict('os.environ', {}, clear=True):
            runner.runtime_url = None
            with pytest.raises(RuntimeError, match='OPENHANDS_RUNTIME_URL'):
                runner.run_parallel([self._make_delegate_action('a', 'b', 'c1')])

    def test_runner_creates_subagent_state(self):
        """Verify _create_subagent_state builds proper state for a subagent."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url='http://test:8000',
        )

        mock_agent = Mock()
        mock_agent.get_system_message.return_value = None

        state = runner._create_subagent_state(
            mock_agent, {'task': 'Read the file', 'description': 'test'}
        )

        assert state.inputs == {'task': 'Read the file', 'description': 'test'}
        assert len(state.history) == 1
        task_msg = state.history[0]
        assert isinstance(task_msg, MessageAction)
        assert 'TASK: Read the file' in task_msg.content

    def test_runner_creates_state_with_system_message(self):
        """State includes system message when agent provides one."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )
        from openhands.events.action.message import SystemMessageAction

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url='http://test:8000',
        )

        sys_msg = SystemMessageAction(content='You are a helpful agent.')
        mock_agent = Mock()
        mock_agent.get_system_message.return_value = sys_msg

        state = runner._create_subagent_state(mock_agent, {'task': 'Do X'})

        assert len(state.history) == 2
        assert isinstance(state.history[0], SystemMessageAction)
        assert isinstance(state.history[1], MessageAction)

    @patch('httpx.Client')
    def test_execute_action_success(self, mock_client_cls):
        """_execute_action makes HTTP POST and returns observation."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'observation': 'read',
            'content': 'file contents here',
            'extras': {},
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_cls.return_value = mock_client

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url='http://localhost:9999',
        )

        action = OpenCodeReadAction(path='/test.py')
        obs = runner._execute_action(action)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert '/execute_action' in call_args[0][0]

    @patch('httpx.Client')
    def test_execute_action_timeout(self, mock_client_cls):
        """_execute_action handles timeout gracefully."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        mock_client = Mock()
        mock_client.post.side_effect = Exception('Timeout')
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_cls.return_value = mock_client

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url='http://localhost:9999',
        )

        action = OpenCodeReadAction(path='/test.py')
        obs = runner._execute_action(action)

        from openhands.events.observation.error import ErrorObservation

        assert isinstance(obs, ErrorObservation)
        assert 'failed' in obs.content.lower() or 'Timeout' in obs.content

    @patch(
        'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner._run_subagent_loop'
    )
    def test_run_parallel_concurrent_execution(self, mock_loop):
        """run_parallel executes multiple subagents and collects results."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        mock_loop.side_effect = [
            {'content': 'Result A', 'outputs': {'content': 'Result A'}},
            {'content': 'Result B', 'outputs': {'content': 'Result B'}},
        ]

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url='http://localhost:9999',
        )

        actions = [
            self._make_delegate_action(
                'OpenCodeGeneralSubAgent', 'Task A', 'call_001'
            ),
            self._make_delegate_action(
                'OpenCodeExploreSubAgent', 'Task B', 'call_002'
            ),
        ]

        results = runner.run_parallel(actions)

        assert len(results) == 2
        assert 'call_001' in results
        assert 'call_002' in results
        result_contents = {results['call_001']['content'], results['call_002']['content']}
        assert result_contents == {'Result A', 'Result B'}

    @patch(
        'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner._run_subagent_loop'
    )
    def test_run_parallel_handles_errors(self, mock_loop):
        """run_parallel handles individual subagent failures gracefully."""
        from openhands.agenthub.opencode_agent.subagents.parallel_runner import (
            ParallelSubagentRunner,
        )

        mock_loop.side_effect = [
            {'content': 'Success', 'outputs': {}},
            RuntimeError('LLM unavailable'),
        ]

        runner = ParallelSubagentRunner(
            llm_registry=Mock(),
            parent_agent_config=Mock(),
            agent_configs={},
            runtime_url='http://localhost:9999',
        )

        actions = [
            self._make_delegate_action('AgentA', 'Task 1', 'call_x'),
            self._make_delegate_action('AgentB', 'Task 2', 'call_y'),
        ]

        results = runner.run_parallel(actions)

        assert len(results) == 2
        assert results['call_x']['content'] == 'Success'
        assert 'error' in results['call_y']['content'].lower()


class TestOpenCodeAgentParallelDetection:
    """Tests that OpenCodeAgent.step() correctly detects and dispatches parallel task calls."""

    def _build_agent(self):
        """Create a mock OpenCodeAgent for testing."""
        from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent

        mock_llm_config = Mock()
        mock_llm_config.model = 'test-model'
        mock_llm_config.max_message_chars = 10000
        mock_llm_config.custom_llm_provider = None

        mock_llm = Mock()
        mock_llm.config = mock_llm_config
        mock_llm.vision_is_active.return_value = False
        mock_llm.is_caching_prompt_active.return_value = False

        mock_registry = Mock()
        mock_registry.get.return_value = mock_llm
        mock_registry.get_router.return_value = mock_llm

        agent_config = AgentConfig()
        agent_config.enable_cmd = False
        agent_config.enable_finish = True

        with patch.object(
            OpenCodeAgent, '__init__', lambda self, *a, **k: None
        ):
            agent = OpenCodeAgent.__new__(OpenCodeAgent)
            agent.config = agent_config
            agent.llm_registry = mock_registry
            agent.llm = mock_llm
            agent.pending_actions = deque()
            agent.tools = []
            agent.mcp_tools = {}
            agent.condenser = Mock()
            agent.conversation_memory = Mock()
            agent.conversation_memory.process_events = Mock(return_value=[])
            agent.conversation_memory.apply_prompt_caching = Mock()
            agent._prompt_manager = Mock()
        return agent

    def _make_multi_task_response(self, num_tasks: int) -> ModelResponse:
        """Create a response with multiple task tool calls."""
        calls = []
        for i in range(num_tasks):
            calls.append(
                {
                    'function': {
                        'name': 'task',
                        'arguments': json.dumps(
                            {
                                'prompt': f'Do task {i}',
                                'subagent_type': 'general',
                            }
                        ),
                    },
                    'id': f'call_{i}',
                    'type': 'function',
                }
            )
        return ModelResponse(
            id='multi-task-resp',
            choices=[
                {
                    'message': {
                        'tool_calls': calls,
                        'content': None,
                        'role': 'assistant',
                    },
                    'index': 0,
                    'finish_reason': 'tool_calls',
                }
            ],
        )

    def test_parallel_detection_with_multiple_task_calls(self):
        """When ALL actions are AgentDelegateAction and count ≥ 2, parallel path triggers."""
        agent = self._build_agent()

        state = State(session_id='test-parallel')
        initial_msg = MessageAction(content='Do multiple things')
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )

        multi_resp = self._make_multi_task_response(3)
        finish_resp = ModelResponse(
            id='finish-resp',
            choices=[
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'finish',
                                    'arguments': json.dumps(
                                        {'message': 'All done'}
                                    ),
                                },
                                'id': 'call_finish',
                                'type': 'function',
                            }
                        ],
                        'content': None,
                        'role': 'assistant',
                    },
                    'index': 0,
                    'finish_reason': 'tool_calls',
                }
            ],
        )

        agent.llm.completion = Mock(side_effect=[multi_resp, finish_resp])

        with patch.dict('os.environ', {'OPENHANDS_RUNTIME_URL': 'http://test:8000'}):
            with patch(
                'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner.run_parallel'
            ) as mock_runner:
                mock_runner.return_value = {
                    'call_0': {'content': 'Result 0', 'outputs': {'content': 'Result 0'}},
                    'call_1': {'content': 'Result 1', 'outputs': {'content': 'Result 1'}},
                    'call_2': {'content': 'Result 2', 'outputs': {'content': 'Result 2'}},
                }

                action = agent.step(state)

                mock_runner.assert_called_once()
                assert agent.llm.completion.call_count == 2

    def test_sequential_fallback_for_single_task(self):
        """A single task tool call does NOT trigger parallel execution."""
        agent = self._build_agent()

        state = State(session_id='test-seq')
        initial_msg = MessageAction(content='Do one thing')
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )

        single_resp = self._make_multi_task_response(1)
        agent.llm.completion = Mock(return_value=single_resp)

        with patch.dict('os.environ', {'OPENHANDS_RUNTIME_URL': 'http://test:8000'}):
            with patch(
                'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner.run_parallel'
            ) as mock_runner:
                action = agent.step(state)
                mock_runner.assert_not_called()
                assert isinstance(action, AgentDelegateAction)

    def test_sequential_fallback_for_mixed_calls(self):
        """Mixed task + non-task calls fall back to sequential."""
        agent = self._build_agent()

        state = State(session_id='test-mixed')
        initial_msg = MessageAction(content='Do things')
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )

        mixed_resp = create_mock_response_multi_tool(
            [
                ('task', {'prompt': 'Explore codebase', 'subagent_type': 'explore'}),
                ('read', {'file_path': '/test.py'}),
            ]
        )
        agent.llm.completion = Mock(return_value=mixed_resp)

        with patch.dict('os.environ', {'OPENHANDS_RUNTIME_URL': 'http://test:8000'}):
            with patch(
                'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner.run_parallel'
            ) as mock_runner:
                action = agent.step(state)
                mock_runner.assert_not_called()

    def test_sequential_fallback_without_runtime_url(self):
        """Without OPENHANDS_RUNTIME_URL, falls back to sequential even with multiple tasks."""
        agent = self._build_agent()

        state = State(session_id='test-no-url')
        initial_msg = MessageAction(content='Do things')
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )

        multi_resp = self._make_multi_task_response(2)
        agent.llm.completion = Mock(return_value=multi_resp)

        with patch.dict('os.environ', {}, clear=True):
            with patch(
                'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner.run_parallel'
            ) as mock_runner:
                action = agent.step(state)
                mock_runner.assert_not_called()
                assert isinstance(action, AgentDelegateAction)

    def test_parallel_injects_synthetic_events(self):
        """After parallel execution, synthetic events are injected into state.history."""
        agent = self._build_agent()

        state = State(session_id='test-inject')
        initial_msg = MessageAction(content='Do two things')
        initial_msg._source = EventSource.USER
        state.history = [initial_msg]
        state.get_last_user_message = Mock(return_value=initial_msg)

        from openhands.memory.condenser.condenser import View

        agent.condenser.condensed_history = Mock(
            return_value=View(events=state.history)
        )

        multi_resp = self._make_multi_task_response(2)
        finish_resp = ModelResponse(
            id='finish-resp-2',
            choices=[
                {
                    'message': {
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'finish',
                                    'arguments': json.dumps(
                                        {'message': 'Done'}
                                    ),
                                },
                                'id': 'call_f',
                                'type': 'function',
                            }
                        ],
                        'content': None,
                        'role': 'assistant',
                    },
                    'index': 0,
                    'finish_reason': 'tool_calls',
                }
            ],
        )
        agent.llm.completion = Mock(side_effect=[multi_resp, finish_resp])

        history_len_before = len(state.history)

        with patch.dict('os.environ', {'OPENHANDS_RUNTIME_URL': 'http://test:8000'}):
            with patch(
                'openhands.agenthub.opencode_agent.subagents.parallel_runner.ParallelSubagentRunner.run_parallel'
            ) as mock_runner:
                mock_runner.return_value = {
                    'call_0': {'content': 'R0', 'outputs': {'content': 'R0'}},
                    'call_1': {'content': 'R1', 'outputs': {'content': 'R1'}},
                }
                action = agent.step(state)

                # 2 tasks -> 2 synthetic actions + 2 synthetic observations = 4 new events
                assert len(state.history) == history_len_before + 4

                synthetic_actions = [
                    e
                    for e in state.history[history_len_before:]
                    if isinstance(e, AgentDelegateAction)
                ]
                synthetic_obs = [
                    e
                    for e in state.history[history_len_before:]
                    if isinstance(e, AgentDelegateObservation)
                ]

                assert len(synthetic_actions) == 2
                assert len(synthetic_obs) == 2

                for obs in synthetic_obs:
                    assert obs.tool_call_metadata is not None
