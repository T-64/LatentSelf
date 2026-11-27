import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.db import connect
from latentself.fact_merge import merge
from latentself.facts import fact_id
from tests.fakes import FakeLLM


class FactMergeTest(unittest.TestCase):
    def test_identity_successive_states_distinct_and_resume(self):
        angles = {"A":0,"alias":0.01,"B":0.54,"C":1.08,"independent":1.6}
        class Embed:
            model = "fixture"
            def embed_texts(self,texts):
                return [[math.cos(angles[t]),math.sin(angles[t])] for t in texts]
        def compare(messages):
            new = json.loads(messages[-1]["content"])["new"]["text"]
            return {"relation":"different" if new=="independent" else "new_replaces_old"}
        llm,backend = FakeLLM(compare),Embed()
        with TemporaryDirectory() as tmp:
            db = connect(Path(tmp)/"db")
            self.addCleanup(db.close)
            for i,text in enumerate(angles,1):
                ts = f"2024-01-0{i}"
                db.execute("INSERT INTO conversation(id,source,first_ts) VALUES (?,'fixture',?)",(text,ts))
                db.execute("INSERT INTO turn VALUES (?,'fixture','prompt',?,?,'s')",(text,ts,text))
                item = dict(text=text,kind="state",date_from=ts,date_to=None,confidence=3 if text=="alias" else 2,
                            evidence={text:text})
                db.execute("INSERT INTO fact_source VALUES (?,'s','fixture',?,NULL)",(text,json.dumps([item])))
            db.commit()
            self.assertEqual(merge(db,llm,backend,vectors_only=True)["vectors_written"],5)
            result = merge(db,llm,backend,emit=lambda _:None)
            self.assertEqual((result["same"],result["new_replaces_old"],result["different"]),(1,2,2))
            chain = {r[0]:tuple(r[1:]) for r in db.execute("SELECT text,status,superseded_by,date_to FROM fact")}
            self.assertEqual(chain["A"],("superseded",fact_id("B"),"2024-01-03"))
            self.assertEqual(chain["B"],("superseded",fact_id("C"),"2024-01-04"))
            self.assertEqual(chain["C"][0],"active")
            self.assertEqual(db.execute("SELECT confidence FROM fact WHERE text='A'").fetchone()[0],3)
            self.assertEqual(db.execute("SELECT count(*) FROM fact_evidence").fetchone()[0],5)
            self.assertEqual(merge(db,llm,backend)["skipped"],5)
