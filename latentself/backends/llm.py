import os

from . import post


class LocalLLM:
    def __init__(self):
        if os.environ.get("LATENTSELF_LLM", "local") != "local":
            raise ValueError("batch inference requires LATENTSELF_LLM=local")
        self.url = os.environ.get("LATENTSELF_LLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = os.environ.get("LATENTSELF_LLM_MODEL", "qwen3.8-27b")

    def chat(self, messages, images=(), json_schema=None, max_tokens=None):
        import json
        messages = list(messages)
        if images:
            content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
            content.append({"type": "text", "text": messages[-1]["content"]})
            messages[-1] = {**messages[-1], "content": content}
        payload = dict(model=self.model, messages=messages, temperature=0.2,
                       chat_template_kwargs={"enable_thinking": False})
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_schema:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "result", "schema": json_schema}}
        choice = post(self.url + "/chat/completions", payload, timeout=300)["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError(f"incomplete model response: output or context limit reached (max_tokens={max_tokens})")
        result = choice["message"]["content"]
        return json.loads(result) if json_schema else result
