import base64
import os

import numpy as np

from . import post

VL = "Qwen/Qwen3-VL-Embedding-8B"
TEXT = "Qwen/Qwen3-Embedding-8B"
INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query."


def normalize(vectors):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("expected a matrix of embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("invalid embedding norm")
    return matrix / norms


class Embedder:
    def __init__(self, model=VL, url=None):
        self.model = model
        if model == TEXT:
            default = os.environ.get("LATENTSELF_TEXT_EMBED_URL", "http://127.0.0.1:8003/v1")
        else:
            default = os.environ.get("LATENTSELF_EMBED_URL", "http://127.0.0.1:8001/v1")
        self.url = (url or default).rstrip("/")
        self.multimodal = model == VL

    def _request(self, **fields):
        rows = post(self.url + "/embeddings", {"model": self.model, "encoding_format": "base64", **fields})["data"]
        return [np.frombuffer(base64.b64decode(r["embedding"],validate=True),dtype="<f4")
                for r in sorted(rows,key=lambda r: r["index"])]

    def _chat(self, text, images, kind):
        content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
        content.append({"type": "text", "text": text})
        return [{"role": "system", "content": INSTRUCTION if kind == "query" else "Represent the user's input."},
                {"role": "user", "content": content}]

    def embed_texts(self, texts, kind="doc"):
        if self.multimodal:
            messages = [self._chat(text, (), kind) for text in texts]
            return normalize(self._request(messages=messages))
        inputs = [f"Instruct: {INSTRUCTION}\nQuery: {t}" if kind == "query" else t for t in texts]
        return normalize(self._request(input=inputs))

    def embed_images(self, items):
        if not self.multimodal:
            raise ValueError("image embeddings require a visual model")
        return normalize(self._request(messages=[self._chat(text, images, "doc") for text, images in items]))


class Reranker:
    model = "Qwen/Qwen3-VL-Reranker-8B"

    def score(self, query, documents):
        base = os.environ.get("LATENTSELF_RERANK_URL", "http://127.0.0.1:8002").rstrip("/")
        rows = post(base + "/rerank", {"model": self.model, "query": query, "documents": documents})["results"]
        return [r["relevance_score"] for r in sorted(rows, key=lambda r: r["index"])]
