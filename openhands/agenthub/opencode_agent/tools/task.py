from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

TASK_TOOL_NAME = 'task'

SUBAGENT_REGISTRY = {
    'general': 'General-purpose agent for researching complex questions and executing multi-step tasks. Use this agent to execute multiple units of work in parallel.',
    'explore': 'Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions.',
}

SUBAGENT_NAME_MAP = {
    'general': 'OpenCodeGeneralSubAgent',
    'explore': 'OpenCodeExploreSubAgent',
}

_TASK_DESCRIPTION_TEMPLATE = """Launch a new agent to handle complex, multistep tasks autonomously.

Available agent types and the tools they have access to:
{agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

When to use the Task tool:
- Complex research or multi-step implementation tasks that benefit from autonomous execution
- Codebase exploration that requires searching across many files
- Tasks that can be parallelized by launching multiple agents concurrently

When NOT to use the Task tool:
- If you want to read a specific file path, use the read tool instead
- If you are searching for a specific class definition like "class Foo", use the glob tool instead
- If you are searching for code within a specific file or set of 2-3 files, use the read tool instead
- Simple single-step tasks that you can handle directly

Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses.
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. Your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final message.
4. The agent's outputs should generally be trusted.
5. Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, etc.), since it is not aware of the user's intent."""


def _build_task_description() -> str:
    agents_text = '\n'.join(
        f'- {name}: {desc}' for name, desc in SUBAGENT_REGISTRY.items()
    )
    return _TASK_DESCRIPTION_TEMPLATE.format(agents=agents_text)


TaskTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=TASK_TOOL_NAME,
        description=_build_task_description(),
        parameters={
            'type': 'object',
            'required': ['description', 'prompt', 'subagent_type'],
            'properties': {
                'description': {
                    'type': 'string',
                    'description': 'A short (3-5 words) description of the task',
                },
                'prompt': {
                    'type': 'string',
                    'description': 'The task for the agent to perform. Include all necessary context.',
                },
                'subagent_type': {
                    'type': 'string',
                    'enum': list(SUBAGENT_REGISTRY.keys()),
                    'description': 'The type of specialized agent to use for this task',
                },
            },
        },
    ),
)
