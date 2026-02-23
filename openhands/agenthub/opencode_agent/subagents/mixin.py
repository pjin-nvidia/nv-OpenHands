"""Mixin that enriches AgentFinishAction outputs to match the opencode subagent output format.

The opencode task tool returns:
{
    title: "<description>",
    metadata: {
        summary: [{ tool, status, title }],
        sessionId: "<id>",
        model: "<model>"
    },
    output: "<text>\n\n<task_metadata>\nsession_id: <id>\n</task_metadata>"
}

This mixin intercepts AgentFinishAction in step() and populates outputs accordingly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, AgentFinishAction
from openhands.events.observation.observation import Observation

if TYPE_CHECKING:
    from openhands.controller.state.state import State


class SubagentMixin:
    """Mixin for OpenCode subagents that formats outputs to match the opencode task tool format."""

    def _collect_tool_summary(self, state: 'State') -> list[dict]:
        """Scan state.history for tool actions and their observations to build an opencode-style summary."""
        summary: list[dict] = []
        for event in state.history:
            if not hasattr(event, 'tool_call_metadata') or event.tool_call_metadata is None:
                continue
            if isinstance(event, Observation):
                continue

            meta = event.tool_call_metadata
            if meta.function_name is None:
                continue

            entry: dict = {
                'id': meta.tool_call_id or '',
                'tool': meta.function_name,
                'state': {
                    'status': 'completed',
                },
            }

            title = self._derive_tool_title(event, meta.function_name)
            if title:
                entry['state']['title'] = title

            summary.append(entry)
        return summary

    def _derive_tool_title(self, action: Action, tool_name: str) -> str:
        """Derive a human-readable title for a tool invocation."""
        if tool_name == 'read' and hasattr(action, 'path'):
            return f'Read {action.path}'
        if tool_name == 'write' and hasattr(action, 'path'):
            return f'Write {action.path}'
        if tool_name == 'edit' and hasattr(action, 'path'):
            return f'Edit {action.path}'
        if tool_name == 'glob' and hasattr(action, 'pattern'):
            return f'Glob {action.pattern}'
        if tool_name == 'grep' and hasattr(action, 'pattern'):
            return f'Grep {action.pattern}'
        if tool_name == 'list_dir' and hasattr(action, 'path'):
            return f'List {action.path}'
        if tool_name in ('execute_bash', 'cmd_run') and hasattr(action, 'command'):
            cmd_preview = action.command[:60]
            return f'Bash: {cmd_preview}'
        if tool_name == 'think' and hasattr(action, 'thought'):
            return f'Think: {action.thought[:40]}'
        return ''

    def _enrich_finish_action(
        self, action: AgentFinishAction, state: 'State'
    ) -> AgentFinishAction:
        """Enrich an AgentFinishAction with opencode-format metadata.

        The 'content' key is used by ConversationMemory as the primary text
        shown to the parent agent's LLM (via AgentDelegateObservation).
        The 'output' key matches the opencode task tool return format.
        """
        description = state.inputs.get('description', '')
        session_id = state.session_id or ''

        model_name = ''
        if hasattr(self, 'llm') and hasattr(self.llm, 'config'):
            model_name = self.llm.config.model

        summary = self._collect_tool_summary(state)
        final_text = action.final_thought or action.thought or ''

        output_text = final_text + '\n\n' + '\n'.join([
            '<task_metadata>',
            f'session_id: {session_id}',
            '</task_metadata>',
        ])

        action.outputs = {
            'content': output_text,
            'title': description,
            'metadata': {
                'summary': summary,
                'sessionId': session_id,
                'model': model_name,
            },
            'output': output_text,
        }

        logger.debug(
            f'Subagent enriched finish action: title={description}, '
            f'tools_executed={len(summary)}, model={model_name}'
        )
        return action
