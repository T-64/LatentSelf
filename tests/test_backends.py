import base64
from http.client import IncompleteRead
import io
import unittest
from unittest.mock import patch

import numpy as np

from latentself.backends import post
from latentself.backends.embed import Embedder, TEXT, VL
from latentself.backends.llm import LocalLLM


class BackendTest(unittest.TestCase):
    @patch("latentself.backends.sleep")
    @patch("latentself.backends.build_opener")
    def test_partial_response_retries_once_without_returning_partial_data(self, build, sleep):
        opener = build.return_value
        opener.open.side_effect = [IncompleteRead(b"partial"),io.BytesIO(b'{"complete":true}')]
        self.assertEqual(post("http://127.0.0.1:8001/test",{}),{"complete":True})
        self.assertEqual(opener.open.call_count,2)
        sleep.assert_called_once()
        opener.open.side_effect = [IncompleteRead(b"partial"),IncompleteRead(b"again")]
        with self.assertRaisesRegex(OSError,"complete response"):
            post("http://127.0.0.1:8001/test",{})

    @patch("latentself.backends.llm.post")
    def test_truncated_response_is_never_treated_as_a_verdict(self, request):
        request.return_value = {"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}
        with self.assertRaisesRegex(ValueError,"max_tokens=48"):
            LocalLLM().chat([{"role":"user","content":"fixture"}],json_schema={"type":"object"},max_tokens=48)

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
        encoded = base64.b64encode(np.array([3,4],dtype="<f4").tobytes()).decode("ascii")
        request.return_value = {"data": [{"index": 0, "embedding": encoded}]}
        for model in (TEXT, VL):
            backend = Embedder(model)
            for kind in ("query", "doc"):
                np.testing.assert_allclose(backend.embed_texts(["fixture"],kind), [[0.6,0.8]])
                payload = request.call_args.args[1]
                self.assertEqual(payload["encoding_format"],"base64")
                if model == TEXT:
                    self.assertEqual(payload["input"][0].startswith("Instruct:"),kind == "query")
                else:
                    self.assertEqual(payload["messages"][0][-1]["content"][0]["text"], "fixture")
                    self.assertEqual("retrieve" in payload["messages"][0][0]["content"], kind == "query")

    def test_rejects_remote_url_before_transport(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            post("https://example.com/v1", {})
