"""
Phase 2: Embedding Engine — 使用 vLLM + Qwen3-Embedding-8B 生成向量

读取 sessionized_data.json 中每个 session 的 engine_data.embedding_target，
通过 vLLM 的 encode API 批量向量化，输出 embeddings.npy。

模型优先从 ModelScope 本地缓存加载，否则从 HuggingFace 下载。
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_PATH = ROOT / "data" / "processed" / "sessionized_data.json"
EMBEDDING_PATH = ROOT / "data" / "processed" / "embeddings.npy"

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MODELSCOPE_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-Embedding-8B"
MAX_TOKENS = 40000  # 模型上下文窗口 40960，留余量给特殊 token


def resolve_model_path() -> str:
    """优先使用 ModelScope 本地缓存路径，否则回退到 HuggingFace ID。"""
    if MODELSCOPE_CACHE.exists() and (MODELSCOPE_CACHE / "config.json").exists():
        print(f"  Using local ModelScope cache: {MODELSCOPE_CACHE}")
        return str(MODELSCOPE_CACHE)
    print(f"  Model not found locally, will download from HuggingFace: {MODEL_NAME}")
    return MODEL_NAME


def load_sessions() -> list[dict]:
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_texts(sessions: list[dict], tokenizer) -> list[str]:
    """提取所有 embedding_target 文本，截断超长文本至模型上限。"""
    texts = []
    truncated = 0
    for s in sessions:
        text = s["engine_data"]["embedding_target"]
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_TOKENS:
            text = tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True)
            truncated += 1
        texts.append(text)
    if truncated:
        print(f"  Truncated {truncated} sessions exceeding {MAX_TOKENS} tokens")
    return texts


def run():
    from vllm import LLM

    print("[Phase 2] Loading sessions...")
    sessions = load_sessions()
    print(f"  Sessions loaded: {len(sessions)}")

    print(f"[Phase 2] Resolving model path...")
    model_path = resolve_model_path()

    print(f"[Phase 2] Loading tokenizer for truncation check...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    texts = build_texts(sessions, tokenizer)

    print(f"[Phase 2] Loading model into vLLM...")
    llm = LLM(
        model=model_path,
        runner="pooling",
        convert="embed",
        trust_remote_code=True,
        dtype="bfloat16",
    )

    print("[Phase 2] Encoding embeddings...")
    outputs = llm.embed(texts)

    # 提取向量矩阵
    embeddings = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)
    print(f"[Phase 2] Saved to {EMBEDDING_PATH}")


if __name__ == "__main__":
    run()
