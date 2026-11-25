# mypy: ignore-errors
"""Bridge so upstream VERA-MH runs the puddles helper as its provider.

Only the provider (``puddles-helper``) is bridged, onto the OpenAI-compatible
endpoint from the helper env. Personas and judges use VERA's own clients.
"""
import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from llm_clients.llm_factory import LLMFactory
from llm_clients.llm_interface import JudgeLLM, Role
from utils.conversation_utils import build_langchain_messages
from vera import main

from puddles.eval.helper import render_helper_system


class PuddlesHelperLLM(JudgeLLM):
    """The helper under the policy in PUDDLES_VERA_POLICY, as a VERA provider."""

    def __init__(self, name, role, system_prompt=None):
        super().__init__(name, role, system_prompt)
        from openai import OpenAI

        self.model = os.environ.get("PUDDLES_HELPER_MODEL")
        if not self.model:
            raise ValueError("PUDDLES_HELPER_MODEL is not set")
        base_url = os.environ.get("PUDDLES_HELPER_BASE_URL")
        api_key = os.environ.get("PUDDLES_HELPER_API_KEY") or ("none" if base_url else None)
        self.client = OpenAI(**{k: v for k, v in
                                (("api_key", api_key), ("base_url", base_url)) if v is not None})
        temperature = os.environ.get("PUDDLES_HELPER_TEMPERATURE", "0").strip()
        self.temperature = None if temperature.lower() == "none" else float(temperature)
        if role == Role.PROVIDER:
            self.system_prompt = self._policy_prompt()

    @staticmethod
    def _policy_prompt():
        return render_helper_system(os.environ["PUDDLES_VERA_POLICY"],
                                    os.environ.get("PUDDLES_VERA_POLICY_VERSION", "policy"))

    def set_system_prompt(self, system_prompt):
        self.system_prompt = (self._policy_prompt()
                              if self.role == Role.PROVIDER and os.getenv("PUDDLES_VERA_POLICY")
                              else system_prompt)

    def _system_messages(self):
        return [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

    def _create(self, messages, **kwargs):
        extra = {"temperature": self.temperature} if self.temperature is not None else {}
        return self.client.chat.completions.create(
            model=self.model, messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **extra, **kwargs)

    async def start_conversation(self):
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(self, conversation_history=None):
        if not conversation_history:
            return await self.start_conversation()
        messages = self._system_messages()
        for message in build_langchain_messages(self.role, conversation_history):
            if isinstance(message, (AIMessage, HumanMessage)):
                role = "assistant" if isinstance(message, AIMessage) else "user"
                messages.append({"role": role, "content": message.content})
        if not any(message["role"] == "user" for message in messages):
            messages.append({"role": "user", "content": "Begin the conversation."})
        response = await asyncio.to_thread(self._create, messages, max_tokens=512)
        self._set_response_metadata("openai-compatible", model=self.model)
        return self._post_process_response(response.choices[0].message.content or "")

    async def generate_structured_response(self, message, response_model):
        messages = self._system_messages()
        messages.append({"role": "user", "content": message or ""})
        for _ in range(3):
            try:
                response = await asyncio.to_thread(
                    self._create, messages, response_format=response_model, max_tokens=4096)
                result = response.choices[0].message.parsed
                self._set_response_metadata("openai-compatible", model=self.model)
                return result
            except ValueError:
                continue
        raise ValueError("puddles-helper failed to return valid structured output")


def register():
    original = LLMFactory.create_llm

    def create(model_name, name, role, system_prompt=None, **kwargs):
        if model_name == "puddles-helper":
            return PuddlesHelperLLM(name, role, system_prompt)
        return original(model_name, name, role, system_prompt, **kwargs)

    LLMFactory.create_llm = staticmethod(create)


if __name__ == "__main__":
    register()
    raise SystemExit(main())
