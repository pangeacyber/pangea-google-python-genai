from __future__ import annotations

from typing import Optional, Union

import google.auth
from google import genai
from google.genai._api_client import BaseApiClient
from google.genai.client import AsyncClient, DebugConfig
from google.genai.types import HttpOptions, HttpOptionsDict
from pangea.asyncio.services import AIGuardAsync, RedactAsync
from pangea.services import AIGuard, Redact
from typing_extensions import override

from pangea_google_genai.models import AsyncPangeaModels, PangeaModels

__all__ = ("PangeaClient",)


class PangeaClient(genai.Client):
    @override
    def __init__(
        self,
        *,
        vertexai: Optional[bool] = None,
        api_key: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        debug_config: Optional[DebugConfig] = None,
        http_options: Optional[Union[HttpOptions, HttpOptionsDict]] = None,
        pangea_api_key: str,
        pangea_input_recipe: str = "pangea_prompt_guard",
        pangea_output_recipe: str = "pangea_llm_response_guard",
    ):
        super().__init__(
            vertexai=vertexai,
            api_key=api_key,
            credentials=credentials,
            project=project,
            location=location,
            debug_config=debug_config,
            http_options=http_options,
        )
        self._aio = AsyncPangeaClient(
            self._api_client,
            pangea_api_key=pangea_api_key,
            pangea_input_recipe=pangea_input_recipe,
            pangea_output_recipe=pangea_output_recipe,
        )
        self._models = PangeaModels(
            self._api_client,
            ai_guard_client=AIGuard(token=pangea_api_key),
            redact_client=Redact(token=pangea_api_key),
            pangea_input_recipe=pangea_input_recipe,
            pangea_output_recipe=pangea_output_recipe,
        )


class AsyncPangeaClient(AsyncClient):
    @override
    def __init__(
        self,
        api_client: BaseApiClient,
        *,
        pangea_api_key: str,
        pangea_input_recipe: str = "pangea_prompt_guard",
        pangea_output_recipe: str = "pangea_llm_response_guard",
    ):
        super().__init__(api_client)
        self._models = AsyncPangeaModels(
            self._api_client,
            ai_guard_client=AIGuardAsync(token=pangea_api_key),
            redact_client=RedactAsync(token=pangea_api_key),
            pangea_input_recipe=pangea_input_recipe,
            pangea_output_recipe=pangea_output_recipe,
        )
