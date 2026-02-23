from openhands.agenthub.opencode_agent.opencode_agent import OpenCodeAgent
from openhands.agenthub.opencode_agent.subagents.explore import (
    OpenCodeExploreSubAgent,
)
from openhands.agenthub.opencode_agent.subagents.general import (
    OpenCodeGeneralSubAgent,
)
from openhands.controller.agent import Agent

Agent.register('OpenCodeAgent', OpenCodeAgent)
Agent.register('OpenCodeGeneralSubAgent', OpenCodeGeneralSubAgent)
Agent.register('OpenCodeExploreSubAgent', OpenCodeExploreSubAgent)
