from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from latentself.db import connect
from latentself.film_metadata import enrich


class MetadataTest(unittest.TestCase):
    def test_cached_enrichment_only_sends_public_ids(self):
        with TemporaryDirectory() as tmp:
            con = connect(Path(tmp)/"db")
            self.addCleanup(con.close)
            con.execute("INSERT INTO item(id,domain,imdb_id) VALUES ('private-local','film','tt1234567')")
            api = Mock()
            api.get.side_effect = [dict(movie_results=[dict(id=42)]),dict(overview="a film"),dict(title="片名")]
            self.assertEqual(enrich(con,api)["with_overview"],1)
            self.assertEqual(enrich(con,api)["cached"],1)
            self.assertEqual([c.args[0] for c in api.get.call_args_list],["/find/tt1234567","/movie/42","/movie/42"])
