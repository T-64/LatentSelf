from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.db import connect
from latentself.fact_review import export_facts,review_sample,verify
from tests.fakes import FakeLLM


class FactReviewTest(unittest.TestCase):
    def test_changed_evidence_invalidates_verification_and_review(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            db = connect(out/"fixture")
            self.addCleanup(db.close)
            for tid in ("a","b"):
                db.execute("INSERT INTO turn (id,source,kind,ts,prompt,snapshot) VALUES (?,'fixture','prompt','2024-01-01','I read a book','s')",(tid,))
                db.execute("INSERT INTO turn_conv VALUES (?,'c',0)",(tid,))
                db.execute("INSERT INTO fact(id,text,kind,confidence,status) VALUES (?,'reads','event',2,'active')",(tid,))
                db.execute("INSERT INTO fact_evidence VALUES (?,?,'read a book')",(tid,tid))
            db.commit()
            llm = FakeLLM({"verdict":"supported"})
            self.assertEqual(verify(db,llm,emit=lambda _:None)["success"],2)
            self.assertEqual(verify(db,llm)["skipped"],2)
            path = export_facts(db,out)["path"]
            self.assertIn("[t:a]",Path(path).read_text())
            self.assertIn("gemini.google.com/app/c",Path(path).read_text())
            review = review_sample(db,out,label=True,ask=lambda _:"y",emit=lambda _:None)
            self.assertEqual((review["sample"],review["accuracy"],review["accepted"]),(2,1.0,False))
            db.execute("UPDATE turn SET prompt='something else' WHERE id='a'")
            db.commit()
            result = verify(db,llm,emit=lambda _:None)
            self.assertEqual((result["invalid_quotes"],result["rejected"]),(1,1))
            self.assertEqual(db.execute("SELECT status FROM fact WHERE id='a'").fetchone()[0],"rejected")
