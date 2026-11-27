from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from latentself.db import connect
from latentself.film_state import AliasIndex,detect,feedback,review


class FilmStateTest(unittest.TestCase):
    def test_short_alias_and_feedback_precedence(self):
        self.assertEqual(set(AliasIndex([("one","A"),("two","Sample")]).match("I saw a sample")),{"two"})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = connect(root/"db")
            self.addCleanup(db.close)
            db.execute("INSERT INTO item(id,domain,title) VALUES ('i','film','Sample')")
            db.execute("INSERT INTO item_alias VALUES ('i','Sample')")
            db.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES ('t','fake','prompt','2024','I saw Sample','s')")
            db.commit()
            llm = Mock(model="mock")
            llm.chat.return_value = dict(status="watched",sentiment=1)
            self.assertEqual(detect(db,llm)["success"],1)
            self.assertEqual(detect(db,llm)["skipped"],1)
            self.assertEqual(review(db,root,lambda _:"y")["correct"],1)
            self.assertEqual(feedback(db,"Sample","dismiss")["status"],"dismissed")
            self.assertEqual(db.execute("SELECT kind FROM signal").fetchone()[0],"rec_feedback")
