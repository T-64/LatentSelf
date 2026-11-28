import json
from pathlib import Path
import tempfile
import unittest

from tests.fakes import FakeEmbed,FakeLLM
from latentself.backends.embed import VL,TEXT
from latentself.db import connect
from latentself.embed import index
from latentself.evaluation import evaluation_db
from latentself.photo_review import checklist
from latentself.stratified import allocate,candidate_frame,draw,label_batch,open_review,status
from latentself.stratified_metrics import breakdown,cells,estimate,frame_guard,predictions,report


class StratifiedTest(unittest.TestCase):
    def test_allocation(self):
        groups={"aistudio":["a"],"gemini":[str(i) for i in range(99)]}
        self.assertEqual(allocate(groups,{"0","1"},4),{"aistudio":1,"gemini":3})
        self.assertEqual(allocate({"gemini":["x"]},set(),6),{"gemini":1})
        self.assertEqual(allocate({},set(),4),{})

    def test_six_cells_and_resumable_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); legacy=evaluation_db(p/"legacy.db")
            with legacy:
                legacy.execute("INSERT INTO label VALUES ('demo','gold',1,NULL)")
                legacy.execute("INSERT INTO label VALUES ('demo','auto',1,NULL)")
                legacy.execute("INSERT INTO consensus_label VALUES ('demo','auto')")
            legacy.close()
            db=open_review(p/"sample.db",p/"legacy.db");self.addCleanup(db.close)
            self.assertEqual([r[0] for r in db.execute("SELECT cid FROM decision")],["gold"])
            rows=[{"q":"demo"}]
            for s in ("aistudio","gemini"):
                for t in ("S1","S2","S3"):
                    for i in range(20):
                        cid="gold" if (s,t,i)==("gemini","S3",19) else s+t+str(i)
                        db.execute("INSERT INTO frame VALUES('demo',?,?,?,'[]','<excerpt>','https://example.org')",(cid,s,t))
            db.execute("INSERT INTO review_meta VALUES('sealed',?)",(json.dumps(rows),))
            draw(db,rows); sample=list(map(tuple,db.execute("SELECT * FROM sample ORDER BY position")))
            db.execute("DELETE FROM sample");draw(db,rows)
            self.assertEqual(sample,list(map(tuple,db.execute("SELECT * FROM sample ORDER BY position"))))
            self.assertEqual(status(db,rows)["totals"],dict(pool=120,sample=16,labeled=1,skipped=0,pending=15))
            with self.assertRaisesRegex(ValueError,"incomplete"):
                cells(db,rows)
            choices=iter(["s","y","q"])
            self.assertEqual(label_batch(db,rows,ask=lambda _:next(choices),emit=lambda _:None),2)
            self.assertEqual(status(db,rows)["totals"]["skipped"],1)
            from latentself.stratified import carry_decisions
            q,cid,answer=db.execute("SELECT q,cid,answer FROM decision WHERE answer='y' AND cid!='gold'").fetchone()
            with db:
                db.execute("INSERT INTO delegated_label VALUES(?,?,'test-reviewer',?,'fixture','now')",(q,cid,answer))
            self.assertEqual(status(db,rows)["label_sources"],{"owner":1,"delegated":{"test-reviewer":1}})
            fresh=open_review(p/"fresh.db");self.addCleanup(fresh.close)
            carry_decisions(fresh,p/"sample.db")
            self.assertEqual(fresh.execute("SELECT reviewer FROM delegated_label").fetchone()[0],"test-reviewer")
            checklist(db,rows,p/"review.html")
            self.assertIn("&lt;excerpt&gt;",(p/"review.html").read_text())
            with self.assertRaisesRegex(ValueError,"outside"):
                frame_guard(db,"demo",["unseen"])

    def test_weighting_bootstrap_and_missing_cells(self):
        data={("a","S1","gemini"):(100,[("x",True),("y",False)]),
              ("a","S3","aistudio"):(3,[("z",True)])}
        got=estimate(data,{"a":{"x","y"}})
        self.assertAlmostEqual(got["recall"],50/53)
        self.assertEqual(got["precision"],.5)
        self.assertEqual(got["ci95"],estimate(data,{"a":{"x","y"}})["ci95"])
        self.assertEqual(got["missed"],[("a","z")])
        b=breakdown(data,[{"q":"a"}],{"a":{"x","y"}},"fast")
        self.assertFalse(b["text"]["meets_text_gate"])
        self.assertEqual(b["by_source"]["aistudio"]["recall"],0)
        data["b","S2","gemini"]=(2,[])
        self.assertIsNone(estimate(data,{"a":set(),"b":set()})["precision"])

    def test_photo_picker_saves_target_without_creating_a_label(self):
        from io import BytesIO
        from unittest.mock import patch, MagicMock
        from latentself.photo_review import serve_photos
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            con=connect(p/"core.db"); db=open_review(p/"review.db")
            self.addCleanup(con.close); self.addCleanup(db.close)
            rows=[{"q":"photo demo","photo":True}]
            db.execute("INSERT INTO review_meta VALUES('sealed',?)",(json.dumps(rows),))
            with self.assertRaisesRegex(ValueError,"select the target photos"):
                report(con,db,rows)
            photo=p/"fixture.jpg"; photo.touch()
            con.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES('t','aistudio','prompt','','fixture','test')")
            con.execute("INSERT INTO answer VALUES('t',0,'','example landscape')")
            con.execute("""INSERT INTO attachment(id,turn_id,pos,stored_name,path,match,mime)
                VALUES('p','t',0,'fixture.jpg',?,'exact','image/jpeg')""",(str(photo),))
            with patch("latentself.photo_review.HTTPServer") as server, patch("webbrowser.open"), patch("secrets.token_urlsafe",return_value="demo"):
                serve_photos(con,db,rows,emit=lambda _:None)
            address,handler=server.call_args.args
            self.assertEqual(address,("127.0.0.1",0))
            def request(raw):
                sock=MagicMock()
                sock.makefile.return_value=BytesIO(raw)
                handler(sock,("127.0.0.1",1),MagicMock())
                return b"".join(call.args[0] for call in sock.sendall.call_args_list)
            page=request(b"GET /demo/?source=aistudio HTTP/1.0\r\n\r\n")
            self.assertIn(b"fixture.jpg",page)
            self.assertIn(b'<form method="post">',request(b"GET /demo/?term=FiXtuRe HTTP/1.0\r\n\r\n"))
            self.assertIn(b'<form method="post">',request(b"GET /demo/?term=landscape HTTP/1.0\r\n\r\n"))
            self.assertNotIn(b'<form method="post">',request(b"GET /demo/?term=absent HTTP/1.0\r\n\r\n"))
            self.assertIn(b"404",request(b"GET /unknown/ HTTP/1.0\r\n\r\n"))
            response=request(b"POST /demo/ HTTP/1.0\r\nContent-Length: 7\r\n\r\nq=1&i=0")
            self.assertIn(b"303",response)
            self.assertEqual(db.execute("SELECT path FROM target").fetchone()[0],str(photo))
            self.assertEqual(db.execute("SELECT count(*) FROM decision").fetchone()[0],0)

    def test_complete_frame_and_offline_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); con=connect(p/"corpus.db");db=open_review(p/"sample.db")
            self.addCleanup(con.close);self.addCleanup(db.close)
            for i,s in enumerate(("gemini","aistudio")):
                cid="c"+str(i)
                con.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES(?,?,'prompt','2020','demo','snap')",(cid,s))
                con.execute("INSERT INTO conversation VALUES(?,?,'2020','2020',1,NULL)",(cid,s))
                con.execute("INSERT INTO turn_conv VALUES(?,?,0)",(cid,cid))
                con.execute("INSERT INTO fts_turn VALUES(?,'demo','','')",(cid,))
            con.commit()
            bs=[]
            for m in (VL,TEXT):
                b=FakeEmbed();b.model=m;bs.append(b);index(con,b,include_images=False,emit=lambda _:None)
            def respond(messages):
                if "同义" in messages[0]["content"]:
                    return {"terms":["demo"]}
                batch=json.loads(messages[-1]["content"])["candidates"]
                return {"results":[dict(conv_id=r["conv_id"],relevant=True,reason="fixture") for r in batch]}
            llm=FakeLLM(respond);rows=[{"q":"demo"}]
            candidate_frame(con,db,rows,bs,llm=llm)
            self.assertEqual(status(db,rows)["totals"]["labeled"],0)
            for cfg, in db.execute("SELECT config FROM strategy"):
                frame_guard(db,"demo",predictions(db,rows,cfg)["demo"])
            label_batch(db,rows,ask=lambda _:"y",emit=lambda _:None)
            class Reranker:
                model="fixture"
                def score(self,q,docs):return [.8]*len(docs)
            result=report(con,db,rows,llm=llm,reranker=Reranker())
            self.assertEqual(result["selected"]["text_model"],VL)
            self.assertTrue(result["reports"]["llm"]["text"]["meets_text_gate"])
