import unittest
from unittest.mock import patch

import numpy as np

from latentself.backends import post
from latentself.backends.embed import Embedder, TEXT, VL
from latentself.backends.llm import LocalLLM


class BackendTest(unittest.TestCase):
    @patch("latentself.backends.llm.post")
    def test_structured_vision_request(self, request):
        request.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        messages = [{"role": "user", "content": "describe"}]
        self.assertEqual(LocalLLM().chat(messages, ["data:image/png;base64,eA=="], {"type": "object"}), {"ok": True})
        payload = request.call_args.args[1]
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(len(payload["messages"][0]["content"]), 2)
        self.assertEqual(messages[0]["content"], "describe")

    @patch("latentself.backends.embed.post")
    def test_model_specific_query_and_document_format(self, request):
        request.return_value = {"data": [{"index": 0, "embedding": [3, 4]}]}
        for model in (TEXT, VL):
            backend = Embedder(model)
            for kind in ("query", "doc"):
                np.testing.assert_allclose(backend.embed_texts(["fixture"],kind), [[0.6,0.8]])
                payload = request.call_args.args[1]
                if model == TEXT:
                    self.assertEqual(payload["input"][0].startswith("Instruct:"),kind == "query")
                else:
                    self.assertEqual(payload["messages"][0][-1]["content"][0]["text"], "fixture")
                    self.assertEqual("retrieve" in payload["messages"][0][0]["content"], kind == "query")

    def test_rejects_remote_url_before_transport(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            post("https://example.com/v1", {})
