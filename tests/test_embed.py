import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.fakes import FakeEmbed
from latentself.backends import InferenceHTTPError
from latentself.db import connect
from latentself.embed import index, turn_inputs


class EmbedTest(unittest.TestCase):
    def test_overlapping_requests_commit_each_batch_on_the_database_thread(self):
        rendezvous = threading.Barrier(2)

        class Backend(FakeEmbed):
            def embed_texts(self,texts,kind="doc"):
                rendezvous.wait(timeout=5)
                return super().embed_texts(texts,kind)

        with tempfile.TemporaryDirectory() as folder:
            con = connect(Path(folder)/"db")
            self.addCleanup(con.close)
            con.executemany("INSERT INTO turn VALUES (?,'fixture','prompt','2020',?,'snapshot')",
                            [(str(i),f"fixture {i}") for i in range(40)])
            progress = []
            result = index(con,Backend(),kinds=("turn",),include_images=False,emit=progress.append)
            self.assertEqual(result["success"],40)
            self.assertEqual(progress,["success 32","success 40"])
            self.assertEqual(con.execute("SELECT count(*) FROM embedding").fetchone()[0],40)

    def test_invalid_input_does_not_discard_valid_neighbors(self):
        class Backend(FakeEmbed):
            def embed_texts(self,texts,kind="doc"):
                if "invalid" in texts:
                    raise InferenceHTTPError(400)
                return super().embed_texts(texts,kind)

        with tempfile.TemporaryDirectory() as folder:
            con = connect(Path(folder)/"db")
            self.addCleanup(con.close)
            for key,text in enumerate(("first","invalid","last")):
                con.execute("INSERT INTO turn VALUES (?,'fixture','prompt','2020',?,'snapshot')",(str(key),text))
            backend = Backend()
            def run():
                return index(con,backend,kinds=("turn",),include_images=False,emit=lambda _:None)
            initial = run()
            self.assertEqual((initial["success"],initial["failed"]),(2,1))
            self.assertEqual(initial["failures"][0]["id"],"1")
            con.execute("UPDATE turn SET prompt='corrected' WHERE id='1'")
            resumed = run()
            self.assertEqual((resumed["success"],resumed["skipped"],resumed["failed"]),(1,2,0))

    @patch("latentself.embed.prepare",return_value=(["data:image/jpeg;base64,eA=="],0))
    def test_visual_comparison_indexes_only_selected_turns_and_resumes(self,prepare):
        with tempfile.TemporaryDirectory() as folder:
            con = connect(Path(folder)/"db")
            self.addCleanup(con.close)
            for tid in ("a","b"):
                con.execute("INSERT INTO turn VALUES (?,'gemini','prompt','2020','fixture','s')",(tid,))
                con.execute("""INSERT INTO attachment(id,turn_id,pos,stored_name,path,match,mime,origin)
                    VALUES (?,?,0,'sample.jpg','sample','exact','image/jpeg','user')""",(tid+"-file",tid))
            backend = FakeEmbed()
            first = index(con,backend,kinds=(),turn_ids={"a"},emit=lambda _:None)
            self.assertEqual((first["success"],first["skipped"]),(1,1))
            self.assertEqual(con.execute("SELECT item_id FROM embedding").fetchone()[0],"a-file")
            second = index(con,backend,kinds=(),turn_ids={"a"},emit=lambda _:None)
            self.assertEqual((second["success"],second["skipped"]),(0,2))
            self.assertEqual(prepare.call_count,2)

    def test_incremental_inputs_and_profile_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            con = connect(Path(folder)/"db")
            self.addCleanup(con.close)
            con.executemany("INSERT INTO turn VALUES (?,'gemini','prompt','2020',?,'snapshot')",
                            [('a','x'*2001),('b','fixture')])
            con.execute("INSERT INTO answer VALUES ('a',0,'','reply')")
            con.execute("""INSERT INTO attachment (id,turn_id,pos,stored_name,match,caption,origin)
                        VALUES ('i','a',0,'generated.png','missing','image','generated')""")
            con.commit()
            first = next(turn_inputs(con))
            self.assertIn("reply", first[1])
            self.assertIn("image", first[1])
            self.assertNotIn("reply", first[2])
            self.assertNotIn("image", first[2])
            backend = FakeEmbed()
            def run(**kwargs):
                return index(con,backend,emit=lambda _:None,**kwargs)
            result = run(limit=1)
            self.assertEqual((result["success"],result["limited"],result["omitted_chars"]), (2,2,2))
            self.assertEqual(run(limit=1)["success"],2)
            self.assertEqual(run()["skipped"],4)
            con.execute("UPDATE attachment SET caption='new caption'")
            self.assertEqual(run()["success"],1)
            for row in con.execute("SELECT dim,vec FROM embedding"):
                vec = np.frombuffer(row["vec"],dtype="<f2").astype(np.float32)
                self.assertEqual(len(vec),row["dim"])
                self.assertAlmostEqual(np.linalg.norm(vec),1,places=3)
