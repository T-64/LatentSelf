"""
Phase 5: Latent Space Interpreter — 统计 Probing + LLM Bridge

分析 embedding 空间的结构：
  1. 计算每个 cluster 的 centroid 与全局 centroid 的偏差
  2. 找到标志性维度（deviation 最大的维度）
  3. 提取极端样本（该维度上激活最高/最低的 sessions）
  4. 生成 cluster 对比分析
  5. 将证据包保存为 JSON，供 LLM bridge 和可视化使用
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_PATH = ROOT / "data" / "processed" / "sessionized_data.json"
EMBEDDING_PATH = ROOT / "data" / "processed" / "embeddings.npy"
INTERPRETATION_PATH = ROOT / "data" / "processed" / "interpretations.json"

TOP_K_DIMS = 20       # 每个 cluster 提取的标志性维度数
TOP_K_SAMPLES = 5     # 每个维度方向的极端样本数


def load_data():
    embeddings = np.load(EMBEDDING_PATH)
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    return embeddings, sessions


def cluster_probing(embeddings, sessions):
    """对每个 cluster 做 centroid deviation 分析。"""
    global_centroid = embeddings.mean(axis=0)
    global_std = embeddings.std(axis=0)
    # 避免除零
    global_std = np.where(global_std == 0, 1, global_std)

    cluster_ids = [s["engine_data"]["cluster_id"] for s in sessions]
    unique_clusters = sorted(set(c for c in cluster_ids if c != -1))

    results = {}
    for cid in unique_clusters:
        indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        cluster_embs = embeddings[indices]
        cluster_centroid = cluster_embs.mean(axis=0)

        # Z-score: 该 cluster 偏离全局多少个标准差
        deviation = (cluster_centroid - global_centroid) / global_std

        # 找 deviation 绝对值最大的维度
        top_dim_indices = np.argsort(np.abs(deviation))[-TOP_K_DIMS:][::-1]

        # 对每个 top 维度，找极端样本
        dim_analysis = []
        for dim_idx in top_dim_indices:
            dim_val = float(deviation[dim_idx])
            all_values = embeddings[:, dim_idx]

            # 全局在该维度上最高/最低的 sessions
            high_indices = np.argsort(all_values)[-TOP_K_SAMPLES:][::-1]
            low_indices = np.argsort(all_values)[:TOP_K_SAMPLES]

            high_samples = []
            for idx in high_indices:
                target = sessions[idx]["engine_data"]["embedding_target"]
                high_samples.append({
                    "session_id": sessions[idx]["session_id"],
                    "cluster_id": sessions[idx]["engine_data"]["cluster_id"],
                    "value": float(all_values[idx]),
                    "text": target[:200],
                })

            low_samples = []
            for idx in low_indices:
                target = sessions[idx]["engine_data"]["embedding_target"]
                low_samples.append({
                    "session_id": sessions[idx]["session_id"],
                    "cluster_id": sessions[idx]["engine_data"]["cluster_id"],
                    "value": float(all_values[idx]),
                    "text": target[:200],
                })

            dim_analysis.append({
                "dim": int(dim_idx),
                "z_score": round(dim_val, 3),
                "high_samples": high_samples,
                "low_samples": low_samples,
            })

        # 该 cluster 的代表性 sessions（距离 centroid 最近的）
        distances = np.linalg.norm(cluster_embs - cluster_centroid, axis=1)
        nearest = np.argsort(distances)[:TOP_K_SAMPLES]
        representative = []
        for local_idx in nearest:
            global_idx = indices[local_idx]
            target = sessions[global_idx]["engine_data"]["embedding_target"]
            representative.append({
                "session_id": sessions[global_idx]["session_id"],
                "text": target[:300],
            })

        results[str(cid)] = {
            "cluster_id": cid,
            "size": len(indices),
            "top_dimensions": dim_analysis,
            "representative_sessions": representative,
        }

    return results


def cluster_comparison(embeddings, sessions, cluster_a: int, cluster_b: int):
    """对比两个 cluster 在 embedding 空间中的差异方向。"""
    cluster_ids = [s["engine_data"]["cluster_id"] for s in sessions]

    idx_a = [i for i, c in enumerate(cluster_ids) if c == cluster_a]
    idx_b = [i for i, c in enumerate(cluster_ids) if c == cluster_b]

    centroid_a = embeddings[idx_a].mean(axis=0)
    centroid_b = embeddings[idx_b].mean(axis=0)

    direction = centroid_a - centroid_b
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

    # 所有 sessions 沿该方向的投影
    projections = embeddings @ direction_norm

    # 该方向上的极端样本
    top_a_side = np.argsort(projections)[-TOP_K_SAMPLES:][::-1]
    top_b_side = np.argsort(projections)[:TOP_K_SAMPLES]

    # 最区分的维度
    top_dims = np.argsort(np.abs(direction))[-10:][::-1]

    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "distinguishing_dims": [
            {"dim": int(d), "diff": round(float(direction[d]), 4)}
            for d in top_dims
        ],
        "a_side_samples": [
            {"session_id": sessions[i]["session_id"],
             "cluster_id": sessions[i]["engine_data"]["cluster_id"],
             "projection": round(float(projections[i]), 4),
             "text": sessions[i]["engine_data"]["embedding_target"][:200]}
            for i in top_a_side
        ],
        "b_side_samples": [
            {"session_id": sessions[i]["session_id"],
             "cluster_id": sessions[i]["engine_data"]["cluster_id"],
             "projection": round(float(projections[i]), 4),
             "text": sessions[i]["engine_data"]["embedding_target"][:200]}
            for i in top_b_side
        ],
    }


def build_llm_prompt(cluster_info: dict) -> str:
    """为单个 cluster 构建 LLM 解释 prompt。"""
    cid = cluster_info["cluster_id"]
    size = cluster_info["size"]

    # 代表性 sessions 的文本
    rep_texts = [s["text"] for s in cluster_info["representative_sessions"]]

    # top 维度上 high/low 的对比
    dim_evidence = []
    for dim in cluster_info["top_dimensions"][:5]:
        high_texts = [s["text"][:100] for s in dim["high_samples"][:3]]
        low_texts = [s["text"][:100] for s in dim["low_samples"][:3]]
        dim_evidence.append(
            f"维度 {dim['dim']} (z-score={dim['z_score']}):\n"
            f"  高激活: {' | '.join(high_texts)}\n"
            f"  低激活: {' | '.join(low_texts)}"
        )

    prompt = f"""你是一个 embedding space 分析专家。以下是一个聚类的分析数据。

## Cluster {cid} ({size} sessions)

### 代表性对话 (最靠近聚类中心):
{chr(10).join(f'- {t}' for t in rep_texts)}

### 向量空间证据 (该聚类偏离全局最大的维度):
{chr(10).join(dim_evidence)}

请分析:
1. 给这个 cluster 一个简短的名字 (3-6个字，中文)
2. 解释为什么 embedding 空间把这些 session 放在一起。不要只描述主题，要深入到更本质的共性。
   比如: 不是"关于学习"，而是"用户在系统性地建立对陌生领域的认知框架"。
3. 向量空间的证据 (高/低激活样本的对比) 支持你的解释吗？

请用以下格式回答:
名称: <3-6个字>
解释: <1-3句话的深度解释>
"""
    return prompt


def run():
    print("[Phase 5] Loading data...")
    embeddings, sessions = load_data()
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Sessions: {len(sessions)}")

    print("[Phase 5] Running cluster probing...")
    cluster_results = cluster_probing(embeddings, sessions)
    print(f"  Analyzed {len(cluster_results)} clusters")

    # 生成几个关键对比
    print("[Phase 5] Running cluster comparisons...")
    cluster_ids = sorted(int(k) for k in cluster_results.keys())
    # 选最大的 3 个 cluster 做两两对比
    top3 = sorted(cluster_ids, key=lambda c: cluster_results[str(c)]["size"], reverse=True)[:3]
    comparisons = []
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            comp = cluster_comparison(embeddings, sessions, top3[i], top3[j])
            comparisons.append(comp)
            print(f"  Compared cluster {top3[i]} vs {top3[j]}")

    # 生成 LLM prompts
    print("[Phase 5] Generating LLM prompts...")
    llm_prompts = {}
    for cid_str, info in cluster_results.items():
        llm_prompts[cid_str] = build_llm_prompt(info)

    # 保存结果
    output = {
        "cluster_profiles": cluster_results,
        "comparisons": comparisons,
        "llm_prompts": llm_prompts,
    }

    INTERPRETATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERPRETATION_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Phase 5] Saved to {INTERPRETATION_PATH}")

    # 打印摘要
    print("\n=== Cluster Summary ===")
    for cid_str in sorted(cluster_results.keys(), key=lambda x: int(x)):
        info = cluster_results[cid_str]
        top_dim = info["top_dimensions"][0]
        print(f"  Cluster {cid_str:>2} ({info['size']:>3} sessions): "
              f"top dim={top_dim['dim']}, z={top_dim['z_score']:+.2f}")


if __name__ == "__main__":
    run()
