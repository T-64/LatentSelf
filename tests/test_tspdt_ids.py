import unittest

from latentself.tspdt_ids import match_public_ids,signature


class PublicIdsTest(unittest.TestCase):
    def test_rank_year_title_and_director_must_agree(self):
        row = dict(rank=1,canonicalKey="film:a:2000",rawTitle="A, The",title="The A",creator="Director",year=2000)
        source = {"1":dict(signature=signature("A, The","Director",2000),imdb="tt0000001")}
        self.assertEqual(match_public_ids([row],source),{row["canonicalKey"]:"tt0000001"})
        self.assertEqual(signature("A","Director","1988-98"),signature("A","Director",1988))
        for change in ({"year":1999},{"rank":2},{"creator":"Another"},{"title":"Else","rawTitle":"Else"}):
            self.assertEqual(match_public_ids([dict(row,**change)],source),{})
