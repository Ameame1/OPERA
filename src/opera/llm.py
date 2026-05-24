from __future__ import annotations

import time
from typing import Dict, List, Optional


Message = Dict[str, str]


class ChatLLM:
    def generate(
        self,
        messages: List[Message],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(ChatLLM):
    """Small client for vLLM/OpenAI-compatible chat completions."""

    def __init__(self, base_url: str, model: Optional[str] = None, timeout: int = 180):
        import requests

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout
        self.model = model or self._discover_model()

    def _discover_model(self) -> str:
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                return data[0]["id"]
        except Exception:
            pass
        return "default"

    def generate(
        self,
        messages: List[Message],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        last_error: Optional[Exception] = None
        delay = 1.0
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error


class TransformersLLM(ChatLLM):
    """Local Hugging Face chat model forced onto the selected CUDA device."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        max_length: int = 8192,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype_arg = torch_dtype
        if torch_dtype == "float16":
            dtype_arg = torch.float16
        elif torch_dtype == "bfloat16":
            dtype_arg = torch.bfloat16

        if device.startswith("cuda"):
            device_map = {"": device}
        else:
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_arg,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if device_map is None:
            self.model.to(device)
        self.model.eval()
        self.torch = torch

    def _format_messages(self, messages: List[Message]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        chunks = []
        for message in messages:
            chunks.append(f"{message['role'].upper()}: {message['content']}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    def generate(
        self,
        messages: List[Message],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> str:
        text = self._format_messages(messages)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with self.torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        decoded = self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True,
        )
        return decoded.strip()
