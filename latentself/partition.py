"""Standalone Leiden worker. Vertex positions are local to a single request."""
import json
import sys


def leiden(n,edges,resolution,seed):
    import igraph
    import leidenalg
    if not edges:
        return list(range(n))
    graph = igraph.Graph(n=n,edges=[(i,j) for i,j,_ in edges],directed=False)
    partition = leidenalg.find_partition(graph,leidenalg.RBConfigurationVertexPartition,
        weights=[w for _,_,w in edges],resolution_parameter=resolution,seed=seed,n_iterations=-1)
    return list(partition.membership)


if __name__=="__main__":
    request = json.load(sys.stdin)
    partitions = [leiden(request["n"],request["edges"],resolution,request["seed"])
                  for resolution in request["resolutions"]]
    json.dump(partitions,sys.stdout)
