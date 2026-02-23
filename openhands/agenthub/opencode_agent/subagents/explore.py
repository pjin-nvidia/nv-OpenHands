"""Explore subagent for fast, read-only codebase exploration.

Mirrors opencode's "explore" subagent: restricted to read-only tools
(grep, glob, list, bash, read). No write, edit, todo, or task tools.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent
from openhands.agenthub.opencode_agent.subagents.mixin import SubagentMixin
from openhands.agenthub.opencode_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.opencode_agent.tools.finish import FinishTool
from openhands.agenthub.opencode_agent.tools.glob import GlobTool
from openhands.agenthub.opencode_agent.tools.grep import GrepTool
from openhands.agenthub.opencode_agent.tools.list_dir import ListDirTool
from openhands.agenthub.opencode_agent.tools.read import ReadTool
from openhands.agenthub.opencode_agent.tools.think import ThinkTool
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import AgentFinishAction
from openhands.utils.prompt import PromptManager

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action


class OpenCodeExploreSubAgent(SubagentMixin, OpenCodeAgent):
    """Read-only exploration subagent.

    Matches opencode's explore subagent permissions:
    - Read-only tools: read, glob, grep, list_dir
    - Bash (for read-only commands like git log, find, etc.)
    - Think and finish
    - NO write, edit, todo_read, todo_write, or task
    """

    VERSION = '1.0'

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
            self._prompt_manager = PromptManager(
                prompt_dir=prompt_dir,
                system_prompt_filename='explore_system_prompt.j2',
            )
        return self._prompt_manager

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        tools: list['ChatCompletionToolParam'] = []

        tools.append(ReadTool)
        tools.append(GlobTool)
        tools.append(GrepTool)
        tools.append(ListDirTool)
        tools.append(ThinkTool)

        if self.config.enable_cmd:
            use_short_desc = any(
                substr in self.llm.config.model
                for substr in ['gpt-4', 'o3', 'o1', 'o4']
            )
            tools.append(create_cmd_run_tool(use_short_description=use_short_desc))

        tools.append(FinishTool)
        return tools

    def step(self, state: State) -> 'Action':
        action = super().step(state)

        if isinstance(action, AgentFinishAction):
            action = self._enrich_finish_action(action, state)
            logger.debug(
                f'ExploreSubAgent finishing with {len(action.outputs.get("metadata", {}).get("summary", []))} tool executions'
            )

        return action
