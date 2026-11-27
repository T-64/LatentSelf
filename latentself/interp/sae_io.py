"""Qwen-Scope's residual SAE conventions and global token indexing."""
import numpy as np


def chunks(length,window=1024,overlap=64):
    if not 0<=overlap<window:
        raise ValueError("overlap must be smaller than the window")
    start = 0
    while start<length:
        end = min(start+window,length)
        yield start,end
        if end==length:
            break
        start = end-overlap


def topk_sae(residual,weights,k,formula="card"):
    """NumPy reference implementation for testing the two encoding formulas."""
    if formula not in ("card","centered") or not 0<k<=len(weights["b_enc"]):
        raise ValueError("invalid SAE configuration")
    x = residual if formula=="card" else residual-weights["b_dec"]
    pre = x@weights["W_enc"].T+weights["b_enc"]
    if formula=="centered":
        pre = np.maximum(pre,0)
    indices = np.argsort(-pre,axis=1,kind="stable")[:,:k]
    values = np.take_along_axis(pre,indices,axis=1)
    sparse = np.zeros_like(pre)
    np.put_along_axis(sparse,indices,values,axis=1)
    reconstructed = sparse@weights["W_dec"].T+weights["b_dec"]
    return indices,values,reconstructed


class Pool:
    """Retain the strongest positive activation and its original token offset."""
    def __init__(self):
        self.best = {}

    def add(self,start,indices,values):
        for offset,(features,activations) in enumerate(zip(indices,values,strict=True),start):
            if offset==0:
                continue
            for feature,value in zip(features,activations,strict=True):
                feature,value = int(feature),float(value)
                previous = self.best.get(feature)
                if value>0 and (previous is None or value>previous[0] or value==previous[0] and offset<previous[1]):
                    self.best[feature] = (value,offset)

    def strongest(self,limit=128):
        return [(feature,act,offset) for feature,(act,offset) in
                sorted(self.best.items(),key=lambda item:(-item[1][0],item[0]))[:limit]]
