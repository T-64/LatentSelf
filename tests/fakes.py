"""Deterministic inference doubles for the offline suite."""
import hashlib

from latentself.backends.embed import normalize


class FakeLLM:
    model = "synthetic-llm"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, images=(), json_schema=None, max_tokens=None):
        self.calls.append((messages, images, json_schema))
        return self.reply(messages) if callable(self.reply) else self.reply


class FakeEmbed:
    model = "synthetic-embedding"
    multimodal = True

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts, kind="doc"):
        self.calls.append((kind, len(texts)))
        return normalize([list(hashlib.sha256(text.encode()).digest()) for text in texts])

    def embed_images(self, items):
        return self.embed_texts([text + "".join(images) for text, images in items])
