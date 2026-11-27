import unittest

from latentself.evidence import cited_lines,original_quote,references


class EvidenceTest(unittest.TestCase):
    def test_whitespace_and_provenance(self):
        self.assertEqual(original_quote("原话是 看\n 书。","看书"),"看\n 书")
        self.assertIsNone(original_quote("我只说过看书。","我喜欢电影。"))
        for q in ("","  ","x"*121):
            self.assertIsNone(original_quote(q,q))

    def test_only_supported_claims_remain(self):
        md = "## 事件\n- 阅读 [c:a] [c:wrong]\n- 购物 [t:elsewhere]\n- 推测"
        body,rejected,omitted = cited_lines(md,{"c:a"})
        self.assertEqual(body,"## 事件\n- 阅读 [c:a]")
        self.assertEqual((rejected,omitted),(2,2))
        self.assertEqual(references(body),{"c:a"})
