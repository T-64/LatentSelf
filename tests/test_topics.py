import importlib.util
import unittest
import json
from unittest.mock import patch,Mock
import numpy as np

from latentself.topics import adjusted_rand,leiden,mutual_knn,recurrence,remote_partitions


class TopicTest(unittest.TestCase):
    def test_remote_worker_accepts_only_graph_data(self):
        with patch("latentself.topics.subprocess.run",return_value=Mock(stdout="[[0,0]]")) as run:
            self.assertEqual(remote_partitions(2,[(0,1,.8)],[1.],42),[[0,0]])
            self.assertEqual(set(json.loads(run.call_args.kwargs["input"])),{"n","edges","resolutions","seed"})

    def test_mutual_neighbours_and_recurrence(self):
        x = np.array([[1,0],[.99,.01],[0,1],[.01,.99]])
        edges = mutual_knn(x,1)
        self.assertEqual([(i,j) for i,j,_ in edges],[(0,1),(2,3)])
        self.assertEqual(recurrence(["2024-01-01","2024-01-15","2024-01-30","2024-01-30"]),(1,3))
        self.assertEqual(adjusted_rand([1,1,0,0],[0,0,1,1]),1.)
        self.assertAlmostEqual(adjusted_rand([1,1,0,0],[0,1,0,1]),-.5)

    @unittest.skipUnless(importlib.util.find_spec("leidenalg"),"optional Leiden runtime")
    def test_leiden_finds_disconnected_cliques(self):
        groups = leiden(4,[(0,1,1),(2,3,1)],1.,42)
        self.assertEqual(adjusted_rand([0,0,1,1],groups),1.)
