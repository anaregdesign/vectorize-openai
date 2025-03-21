import os
from typing import Type, TypeVar

import pandas as pd
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from openaivec.embedding import EmbeddingOpenAI
from openaivec.vectorize import VectorizedLLM, VectorizedOpenAI

__all__ = [
    "use_openai",
    "use_azure_openai",
]


T = TypeVar("T")

_client: OpenAI | None = None


def use_openai(api_key: str) -> None:
    """
    Set the OpenAI API key to use for OpenAI and Azure OpenAI.
    """
    global _client
    _client = OpenAI(api_key=api_key)


def use_azure_openai(api_key: str, endpoint: str, api_version: str) -> None:
    """
    Set the Azure OpenAI API key to use for Azure OpenAI.
    """
    global _client
    _client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_openai_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    if "OPENAI_API_KEY" in os.environ:
        _client = OpenAI()
        return _client

    aoai_param_names = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ]

    if all(param in os.environ for param in aoai_param_names):
        _client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

        return _client

    raise ValueError(
        "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable or provide Azure OpenAI parameters."
        "If using Azure OpenAI, ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION are set."
        "If using OpenAI, ensure OPENAI_API_KEY is set."
    )


@pd.api.extensions.register_series_accessor("ai")
class OpenAIVecSeriesAccessor:
    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def predict(self, model_name: str, prompt: str, respnse_format: Type[T] = str, batch_size: int = 128) -> pd.Series:
        client: VectorizedLLM = VectorizedOpenAI(
            client=get_openai_client(),
            model_name=model_name,
            system_message=prompt,
            is_parallel=True,
            response_format=respnse_format,
            temperature=0,
            top_p=1,
        )

        return pd.Series(
            client.predict_minibatch(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
        )

    def embed(self, model_name: str, batch_size: int = 128) -> pd.Series:
        client: VectorizedLLM = EmbeddingOpenAI(
            client=get_openai_client(),
            model_name=model_name,
        )

        return pd.Series(
            client.embed_minibatch(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
        )

    def extract(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._obj.map(lambda x: x.model_dump() if isinstance(x, BaseModel) else {self._obj.name: x}).tolist(),
            index=self._obj.index,
        )
