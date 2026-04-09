"""
Phase 3: Manifold & Clustering — UMAP 降维 + HDBSCAN 聚类

读取 embeddings.npy，UMAP 投影到 3D，HDBSCAN 密度聚类，
将 umap_coords 和 cluster_id 回写到 sessionized_data.json。
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_PATH = ROOT / "data" / "processed" / "sessionized_data.json"
EMBEDDING_PATH = ROOT / "data" / "processed" / "embeddings.npy"


def run():
    import umap
    import hdbscan

    # ── 加载数据 ──
    print("[Phase 3] Loading data...")
    embeddings = np.load(EMBEDDING_PATH)
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    assert len(sessions) == embeddings.shape[0], (
        f"Mismatch: {len(sessions)} sessions vs {embeddings.shape[0]} embeddings"
    )
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Sessions:   {len(sessions)}")

    # ── UMAP 降维 ──
    print("[Phase 3] Running UMAP (4096 → 3D, cosine)...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_3d = reducer.fit_transform(embeddings)
    print(f"  UMAP output: {coords_3d.shape}")

    # ── HDBSCAN 聚类 ──
    print("[Phase 3] Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords_3d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

    # 各簇大小
    for cid in sorted(set(labels)):
        count = int(np.sum(labels == cid))
        tag = "noise" if cid == -1 else f"cluster {cid}"
        print(f"    {tag}: {count}")

    # ── 回写 JSON ──
    print("[Phase 3] Writing back to sessionized_data.json...")
    for i, session in enumerate(sessions):
        session["engine_data"]["umap_coords"] = coords_3d[i].tolist()
        session["engine_data"]["cluster_id"] = int(labels[i])

    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

    print(f"[Phase 3] Done. Updated {len(sessions)} sessions.")


if __name__ == "__main__":
    run()
