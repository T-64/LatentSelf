import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from latentself.db import connect
from latentself.read_api import Reader,bounded,readonly
from latentself.search import remember


class ReaderTest(unittest.TestCase):
    def test_bounded_payload_preserves_source_and_counts_omissions(self):
        rows = [dict(turn_id=str(i),url="source",prompt="x"*5000) for i in range(50)]
        result = bounded(rows,cap=500)
        self.assertLessEqual(len(json.dumps(result,ensure_ascii=False)),500)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["omitted"],49)
        self.assertEqual(result["results"][0]["turn_id"],"0")
        self.assertGreater(result["omitted_chars"],4500)

    def test_tools_use_source_ids_links_and_read_only_connections(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"fixture"
            db = connect(path)
            db.execute("INSERT INTO turn (id,source,kind,ts,prompt,snapshot) VALUES ('t','fixture','prompt','2024-03-02','reading','s')")
            db.execute("INSERT INTO answer VALUES ('t',0,'',?)",("x"*24000,))
            db.execute("INSERT INTO conversation(id,source,title) VALUES ('c','fixture','学习')")
            db.execute("INSERT INTO turn_conv VALUES ('t','c',0)")
            db.execute("INSERT INTO conv_summary(conv_id,summary,kinds,model,input_hash) VALUES ('c','摘要','[\"学习\"]','f','h')")
            db.execute("INSERT INTO digest(period,level,body_md,model,input_hash) VALUES ('2024','year','reading [c:c]','f','h')")
            db.execute("INSERT INTO fact(id,text,kind,confidence,status,verified) VALUES ('f','reads','event',3,'active',1)")
            db.execute("INSERT INTO fact_evidence VALUES ('f','t','reading')")
            db.execute("INSERT INTO attachment(id,turn_id,pos,stored_name,match) VALUES ('a','t',0,'image.png','missing')")
            db.commit()
            db.close()
            view = Reader(path)
            self.assertEqual(view.timeline("2024-03","2024-03",["学习"])["results"][0]["url"],"https://gemini.google.com/app/c")
            self.assertFalse(view.timeline("2024-04","2024-05")["results"])
            self.assertEqual(view.digest("2024")["results"][0]["sources"][0]["conv_id"],"c")
            self.assertEqual(view.attachment("a")["results"][0]["turn_id"],"t")
            self.assertEqual(view.facts("read")["results"][0]["evidence"][0]["quote"],"reading")
            self.assertTrue(view.stats("overview")["available"])
            self.assertFalse(view.stats("profile")["available"])
            self.assertGreater(view.show_conversation("c")["omitted_chars"],0)
            self.assertLessEqual(len(json.dumps(view.show_conversation("c",True),ensure_ascii=False)),20000)
            with patch("latentself.read_api.retrieve",return_value={"mode":"fast","results":[{"conv_id":"c"}]}):
                self.assertEqual(view.search("test",limit=0)["omitted"],1)
            ro = readonly(path)
            self.addCleanup(ro.close)
            remember(ro,"blocked","judge",{})
            with self.assertRaises(sqlite3.OperationalError):
                ro.execute("UPDATE fact SET verified=0")
            self.assertEqual(ro.execute("SELECT count(*) FROM search_cache").fetchone()[0],0)
