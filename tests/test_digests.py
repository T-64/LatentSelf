from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from latentself.db import connect
from latentself.backends.llm import OutputLimitExceeded
from latentself.digests import SECTIONS,digests,reduce_material,summarize
from latentself.evidence import references
from tests.fakes import FakeLLM


class DigestTest(unittest.TestCase):
    def test_retry_incomplete_digest_and_restore_citations_from_the_complete_response(self):
        llm = Mock()
        reply = {s:[] for s in SECTIONS}
        reply["事件"] = ["有据记录 [c:1]"]
        llm.chat.side_effect = [OutputLimitExceeded(),reply]
        stats = dict(invalid_citations=0,uncited_points=0)
        body = summarize(llm,"2024",["第一条 [c:original]"],{"c:original"},stats)
        self.assertEqual(references(body),{"c:original"})
        self.assertEqual([c.kwargs["max_tokens"] for c in llm.chat.call_args_list],[2048,4096])
        self.assertEqual(stats["output_limit_retries"],1)

    def test_oversized_month_uses_weekly_sources(self):
        def answer(messages):
            refs = references(messages[-1]["content"])
            return {s:["记录 "+" ".join(f"[{r}]" for r in sorted(refs))] if s=="事件" else [] for s in SECTIONS}
        stats = dict(intermediate_summaries=0,invalid_citations=0,uncited_points=0)
        entries = [(week,"x"*270+f" [c:{week}]",{f"c:{week}"}) for week in (1,2,3)]
        body = reduce_material(FakeLLM(answer),"2024-01",entries,500,stats)
        self.assertEqual(stats["intermediate_summaries"],3)
        self.assertEqual(references(body),{"c:1","c:2","c:3"})

    def test_rollup_validates_period_and_reuses_unchanged_sources(self):
        def answer(messages):
            refs = sorted(references(messages[-1]["content"]))
            self.assertTrue(all(r[2:].isdecimal() for r in refs))
            response = {s:[] for s in SECTIONS}
            response["事件"] = ["合成记录 "+" ".join(f"[{r}]" for r in refs)+" [c:wrong]","无证据"]
            return response
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            db = connect(out/"fixture")
            self.addCleanup(db.close)
            for month,cid in ((1,"a"),(5,"b")):
                ts = f"2024-{month:02d}-05"
                db.execute("INSERT INTO turn VALUES (?,'fixture','prompt',?,'reading','s')",(cid,ts))
                db.execute("INSERT INTO conversation(id,source,title) VALUES (?,'fixture','标题')",(cid,))
                db.execute("INSERT INTO turn_conv VALUES (?,?,0)",(cid,cid))
                db.execute("""INSERT INTO conv_summary(conv_id,summary,kinds,model,input_hash)
                           VALUES (?,'摘要','[]','fixture','s')""",(cid,))
            db.commit()
            llm = FakeLLM(answer)
            result = digests(db,llm,out,period="2024")
            self.assertEqual((result["success"],result["invalid_citations"],result["uncited_points"]),(5,5,5))
            self.assertEqual(references((out/"digests/year/2024.md").read_text()),{"c:a","c:b"})
            self.assertEqual(references((out/"digests/quarter/2024-Q1.md").read_text()),{"c:a"})
            self.assertEqual(digests(db,llm,out,period="2024")["skipped"],5)
            db.execute("UPDATE conv_summary SET summary='变更' WHERE conv_id='a'")
            db.commit()
            self.assertEqual(digests(db,llm,out,period="2024")["success"],1)
