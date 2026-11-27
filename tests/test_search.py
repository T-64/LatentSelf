import json
import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeEmbed,FakeLLM
from latentself.db import connect
from latentself.embed import index
from latentself.ingest.gemini import ingest
from latentself.search import search,VectorIndex


class SearchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.con = connect(root/"test.db")
        self.addCleanup(self.con.close)
        rows = [{"title":"Prompted green sample " + str(i),"time":f"2020-0{i+1}-01T00:00:00Z",
                 "details":[{"url":"https://gemini.google.com/app/"+cid} for cid in cids]}
                for i,cids in enumerate([["ab","cd"],["ab"],["ef"]])]
        (root/"MyActivity.json").write_text(json.dumps(rows))
        ingest(self.con,root)
        self.embed = FakeEmbed()
        index(self.con,self.embed,emit=lambda _:None)
        self.idx = VectorIndex(self.con,self.embed,("turn",))

    def run_search(self,query="green",**kwargs):
        return search(self.con,query,terms=[],indexes=[self.idx],sort="score",**kwargs)

    def test_union_shared_turns_filters_and_counted_cap(self):
        out = self.run_search(top_k=1,cap=2)
        self.assertEqual(len(out["results"]),2)
        self.assertEqual(out["counts"]["conversations_omitted"],1)
        self.assertEqual(out["counts"]["vector_turns_omitted"],2)
        full = self.run_search(top_k=0)
        self.assertEqual(full["results"][0]["conv_id"],"ab")
        self.assertEqual(full["results"][0]["hits"],2)
        self.assertAlmostEqual(full["results"][0]["score"],1+0.05*__import__("math").log(3))
        self.assertEqual([r["conv_id"] for r in self.run_search(since="2020-02",until="2020-02")["results"]],["ab"])
        self.assertEqual(self.run_search(images_only=True)["results"],[])

    def test_judgments_are_cached_and_invalidate_on_evidence_change(self):
        def judge(messages):
            return {"results":[{"conv_id":c["conv_id"],"relevant":c["conv_id"]!="cd","reason":"fixture"}
                               for c in json.loads(messages[-1]["content"])["candidates"]]}
        llm = FakeLLM(judge)
        for _ in range(2):
            out = self.run_search(mode="strict",llm=llm)
            self.assertEqual({r["conv_id"] for r in out["results"]},{"ab","ef"})
        self.assertEqual(len(llm.calls),1)
        self.con.execute("UPDATE turn SET prompt='changed green' WHERE ts LIKE '2020-02%'")
        self.run_search(mode="strict",llm=llm)
        self.assertEqual(len(llm.calls),2)

    def test_unavailable_backend_returns_labeled_keyword_results(self):
        def unavailable(*args,**kwargs):
            raise ConnectionError("offline")
        self.embed.embed_texts = unavailable
        out = self.run_search(mode="strict",llm=FakeLLM(unavailable))
        self.assertEqual(out["mode"],"unjudged")
        self.assertTrue(any("仅关键词" in n for n in out["notices"]))
        self.assertEqual(len(out["results"]),3)
