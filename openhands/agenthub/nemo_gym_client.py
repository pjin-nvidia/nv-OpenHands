"""Shared NeMo Gym client for making model calls via the NeMo Gym server.

This module provides a reusable client that any agent can use to route
LLM completions through the NeMo Gym infrastructure instead of calling
litellm directly.
"""

import json
import os
import tempfile
import time
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient
from nemo_gym.server_utils import get_response_json, raise_for_status

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.core.message import Message
    from openhands.llm.llm import LLM, ModelResponse


class NemoGymClient:
    """Client that proxies LLM completions through the NeMo Gym server.

    Usage::

        # In agent __init__:
        self.nemo_gym_client = NemoGymClient(self.llm)

        # In agent step (async):
        response = await self.nemo_gym_client.model_call(messages, tools)
    """

    def __init__(self, llm: "LLM") -> None:
        self.ng_server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=get_global_config_dict(),
        )
        self.model_server_cookies = None
        self.llm = llm

    async def model_call(
        self,
        messages: list["Message"],
        tools: "list[ChatCompletionToolParam] | None" = None,
    ) -> "ModelResponse":
        """Make a model call via the NeMo Gym server, with automatic metrics tracking.

        Args:
            messages: Conversation messages (OpenHands Message objects).
            tools: Optional list of tool definitions for function calling.

        Returns:
            A validated ModelResponse from the server.
        """
        start_time = time.time()
        response = await self._post_completion(messages, tools)
        self._update_model_call_time(start_time)
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_completion(
        self,
        messages: list["Message"],
        tools: "list[ChatCompletionToolParam] | None" = None,
    ) -> "ModelResponse":
        from openhands.llm.llm import ModelResponse

        message_dicts = [m.model_dump() for m in messages]

        params: dict = {
            "messages": message_dicts,
            **self.llm._nemo_gym_llm_kwargs,
        }
        if tools:
            params["tools"] = tools

        fields_to_remove = [
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        ]
        last_occurrence_idx_seen = False
        for message in reversed(message_dicts):
            if last_occurrence_idx_seen:
                for field in fields_to_remove:
                    if field in message:
                        del message[field]
            elif all(field in message for field in fields_to_remove):
                last_occurrence_idx_seen = True

        # Measure per-call round-trip latency so it's surfaced in
        # `Metrics.response_latencies` (and therefore in the eval output.jsonl
        # via `get_metrics(state)`), mirroring the litellm path in
        # `openhands/llm/llm.py::_completion`.
        latency_start = time.perf_counter()
        model_response = await self.ng_server_client.post(
            server_name=os.getenv("NEMO_GYM_MODEL_SERVER_NAME"),
            url_path="/v1/chat/completions",
            json=params,
            cookies=self.model_server_cookies,
        )
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)
        latency = time.perf_counter() - latency_start
        self.llm.metrics.add_response_latency(
            latency, model_response_json.get("id", "unknown")
        )
        self.model_server_cookies = model_response.cookies

        response: ModelResponse = ModelResponse.model_validate(model_response_json)

        response_message_dict = model_response_json["choices"][0]["message"]
        provider_specific_fields: dict = {}
        if response_message_dict.get("prompt_token_ids"):
            provider_specific_fields = {
                "prompt_token_ids": response_message_dict["prompt_token_ids"],
                "generation_token_ids": response_message_dict["generation_token_ids"],
                "generation_log_probs": response_message_dict["generation_log_probs"],
            }
            response._provider_specific_fields = provider_specific_fields

        self._log_completion(
            messages, model_response_json, provider_specific_fields, params
        )

        return response

    def _log_completion(
        self,
        messages: list["Message"],
        model_response_json: dict,
        provider_specific_fields: dict,
        params: dict,
    ) -> None:
        log_file = os.path.join(
            self.llm.config.log_completions_folder,
            f"{self.llm.config.model.replace('/', '__')}-{time.time()}.json",
        )
        _d = {
            "messages": [m.model_dump() for m in messages],
            "response": model_response_json,
            "provider_specific_fields": provider_specific_fields,
            "kwargs": {
                k: v for k, v in params.items() if k not in ("messages", "client")
            },
            "timestamp": time.time(),
        }

        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(log_file))
        with os.fdopen(temp_fd, "w") as f:
            f.write(json.dumps(_d))
        os.replace(temp_path, log_file)

    @staticmethod
    def _update_model_call_time(start_time: float) -> None:
        metrics_fpath = os.environ.get("NEMO_GYM_METRICS_FPATH")
        if not metrics_fpath:
            return

        from openhands.nemo_gym_metrics_utils import update_json_metrics_file

        update_json_metrics_file(
            metrics_fpath,
            increments={"total_model_call_time": time.time() - start_time},
            log_prefix="swe_agents",
        )
