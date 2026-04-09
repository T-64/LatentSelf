"""
Phase 1.5: Semantic Session Splitter — 基于 Turn-level Embedding 语义拆分

对 Phase 1 产出的粗 session 做语义二次拆分：
  - 只处理 >5 turns 的 session（短对话大概率是连贯的）
  - 用 3-turn 滑动窗口计算语义相似度（避免单 turn 噪声）
  - 相似度 < threshold (0.3) 处切分

需要 GPU 环境 (vllm-env)。
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_PATH = ROOT / "data" / "processed" / "sessionized_data.json"

SIMILARITY_THRESHOLD = 0.4   # 窗口平滑后的阈值
MIN_TURNS_TO_SPLIT = 6       # 至少 6 turns 才考虑拆分
WINDOW_SIZE = 2              # 滑动窗口大小（左2右2）
TIME_FMT = "%Y-%m-%d %H:%M:%S"


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_split_points(embeddings, threshold, window_size):
    """
    用滑动窗口计算前后块的平均 embedding 相似度。
    在 position i 处：
      left_centroid  = mean(embeddings[max(0, i-window):i])
      right_centroid = mean(embeddings[i:min(n, i+window)])
    如果 similarity(left, right) < threshold → 断裂点
    """
    n = len(embeddings)
    if n < 2:
        return []

    splits = []
    for i in range(1, n):
        left_start = max(0, i - window_size)
        right_end = min(n, i + window_size)

        left_centroid = embeddings[left_start:i].mean(axis=0)
        right_centroid = embeddings[i:right_end].mean(axis=0)

        sim = cosine_similarity(left_centroid, right_centroid)
        if sim < threshold:
            splits.append((i, sim))

    return splits


def rebuild_session(original, dialogues_slice, sub_idx, embedding_turns_slice):
    """从原始 session 的一段 dialogues 构建新的子 session。"""
    from datetime import datetime

    start_time = dialogues_slice[0]["timestamp"]
    end_time = dialogues_slice[-1]["timestamp"]
    t0 = datetime.strptime(start_time, TIME_FMT)
    t1 = datetime.strptime(end_time, TIME_FMT)
    duration = round((t1 - t0).total_seconds() / 60, 1)

    new_dialogues = []
    for i, d in enumerate(dialogues_slice, start=1):
        new_dialogues.append({
            "turn": i,
            "timestamp": d["timestamp"],
            "user_prompt": d["user_prompt"],
            "ai_response": d["ai_response"],
        })

    embedding_target = "\n".join(embedding_turns_slice)
    sid = original["session_id"]
    new_sid = f"{sid}_sub{sub_idx}" if sub_idx > 0 else sid

    return {
        "session_id": new_sid,
        "parent_session_id": original["session_id"],
        "metadata": {
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration,
            "turn_count": len(new_dialogues),
        },
        "engine_data": {
            "embedding_turns": embedding_turns_slice,
            "embedding_target": embedding_target,
            "umap_coords": None,
            "cluster_id": None,
        },
        "ui_data": {
            "dialogues": new_dialogues,
        },
    }


def run():
    from vllm import LLM

    print("[Phase 1.5] Loading sessions...")
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    # 区分需要拆分的和不需要的
    to_split = [s for s in sessions if s["metadata"]["turn_count"] >= MIN_TURNS_TO_SPLIT]
    keep_as_is = [s for s in sessions if s["metadata"]["turn_count"] < MIN_TURNS_TO_SPLIT]
    print(f"  Total: {len(sessions)} | To split (>={MIN_TURNS_TO_SPLIT} turns): {len(to_split)} | Keep: {len(keep_as_is)}")

    if not to_split:
        print("[Phase 1.5] No sessions to split. Done.")
        return

    # 收集所有需要 embedding 的 turn prompts
    all_prompts = []
    prompt_map = []
    for si, s in enumerate(to_split):
        turns = s["engine_data"]["embedding_turns"]
        for ti, prompt in enumerate(turns):
            all_prompts.append(prompt)
            prompt_map.append((si, ti))

    print(f"  Total turn prompts to embed: {len(all_prompts)}")

    # 截断超长 prompt
    model_path = str(Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-Embedding-8B")
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen3-Embedding-8B"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    max_tokens = 40000
    truncated = 0
    for i, text in enumerate(all_prompts):
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            all_prompts[i] = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            truncated += 1
    if truncated:
        print(f"  Truncated {truncated} prompts exceeding {max_tokens} tokens")

    print(f"[Phase 1.5] Loading embedding model...")
    llm = LLM(
        model=model_path,
        runner="pooling",
        convert="embed",
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print(f"[Phase 1.5] Embedding {len(all_prompts)} turn prompts...")
    outputs = llm.embed(all_prompts)
    all_embeddings = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    all_embeddings = all_embeddings / norms

    print(f"  Embeddings shape: {all_embeddings.shape}")

    # 按 session 分组
    session_embeddings = {}
    for (si, ti), emb in zip(prompt_map, all_embeddings):
        if si not in session_embeddings:
            session_embeddings[si] = []
        session_embeddings[si].append(emb)

    # 拆分 + 收集相似度分布
    new_sessions = list(keep_as_is)
    total_splits = 0
    split_details = []
    all_sims = []  # 所有相邻窗口的相似度

    for si, s in enumerate(to_split):
        embs = np.array(session_embeddings[si])
        dialogues = s["ui_data"]["dialogues"]
        turns = s["engine_data"]["embedding_turns"]

        # 计算所有相邻窗口相似度（用于统计）
        n = len(embs)
        for i in range(1, n):
            left_start = max(0, i - WINDOW_SIZE)
            right_end = min(n, i + WINDOW_SIZE)
            left_c = embs[left_start:i].mean(axis=0)
            right_c = embs[i:right_end].mean(axis=0)
            sim = cosine_similarity(left_c, right_c)
            all_sims.append(sim)

        split_points = find_split_points(embs, SIMILARITY_THRESHOLD, WINDOW_SIZE)

        if not split_points:
            new_sessions.append(s)
        else:
            indices = [sp[0] for sp in split_points]
            sims = [sp[1] for sp in split_points]
            boundaries = [0] + indices + [len(dialogues)]
            total_splits += len(indices)

            split_details.append(
                f"  {s['session_id']} ({len(dialogues)} turns) → "
                f"{len(boundaries)-1} sub-sessions (sims: {[f'{s:.2f}' for s in sims]})"
            )

            for sub_idx in range(len(boundaries) - 1):
                start = boundaries[sub_idx]
                end = boundaries[sub_idx + 1]
                sub_session = rebuild_session(
                    s, dialogues[start:end], sub_idx, turns[start:end]
                )
                new_sessions.append(sub_session)

    new_sessions.sort(key=lambda s: s["metadata"]["start_time"])

    # 相似度分布统计
    if all_sims:
        sims_arr = np.array(all_sims)
        print(f"\n[Phase 1.5] Windowed similarity distribution ({len(sims_arr)} pairs):")
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            print(f"  P{pct}: {np.percentile(sims_arr, pct):.3f}")
        print(f"  < 0.3: {(sims_arr < 0.3).sum()} ({(sims_arr < 0.3).mean()*100:.1f}%)")
        print(f"  < 0.4: {(sims_arr < 0.4).sum()} ({(sims_arr < 0.4).mean()*100:.1f}%)")
        print(f"  < 0.5: {(sims_arr < 0.5).sum()} ({(sims_arr < 0.5).mean()*100:.1f}%)")

    print(f"\n[Phase 1.5] Results:")
    print(f"  Original sessions: {len(sessions)}")
    print(f"  Sessions analyzed: {len(to_split)}")
    print(f"  Split points found: {total_splits}")
    print(f"  New sessions: {len(new_sessions)} (+{len(new_sessions) - len(sessions)})")

    if split_details:
        print(f"\n  Split details:")
        for d in split_details[:20]:
            print(d)
        if len(split_details) > 20:
            print(f"  ... and {len(split_details) - 20} more")

    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(new_sessions, f, ensure_ascii=False, indent=2)
    print(f"\n[Phase 1.5] Saved to {SESSION_PATH}")


if __name__ == "__main__":
    run()
