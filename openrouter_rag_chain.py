import os
import json
import requests

# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import HumanMessage, AIMessage
# from langchain.llms.base import LLM
from langchain_core.language_models import LLM
# from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional


class OpenRouterLLM(LLM):
    """Custom LLM for OpenRouter API."""

    model: str = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    max_tokens: int = None
    temperature: float = None

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter API."""

        # Set up the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }

        # # If stop sequences are provided, add them
        # ev lägger vi till detta senare, stödjs inte av Llamacode och
        # llama scout, vill man ha det får man utveckla logik själv

        # if stop:
        #     data["stop"] = stop

        # Make the request
        response = requests.post(
            url=self.api_url,
            headers=headers,
            data=json.dumps(data)
        )

        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response from Model {response_json}")
            return response_json["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(
                f"Error {response.status_code}: {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
