"""
Phase 1: Data Pipeline — 基于时间阈值切分 Session

读取 data/raw/MyActivity.json，过滤噪声，按时间间隔将连续的提问合并为 Session，
输出 data/processed/sessionized_data.json。

领域分离 (Separation of Concerns)：
  - metadata:    时间轴过滤和基础统计
  - engine_data: Qwen 向量化 + HDBSCAN 读写
  - ui_data:     Streamlit/Plotly 前端渲染
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = ROOT / "data" / "raw" / "MyActivity.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "sessionized_data.json"

SESSION_GAP_THRESHOLD = 15 * 60  # 15 分钟
TIME_FMT = "%Y-%m-%d %H:%M:%S"
SENSITIVE_PATTERNS = {"a sensitive query", "a sensitive query."}


def load_raw_data() -> list[dict]:
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


def extract_record(record: dict) -> dict | None:
    """从单条原始记录提取 prompt、response、time。"""
    title = record.get("title", "")
    prompt = None

    if title.startswith("Prompted "):
        prompt = title[len("Prompted "):].strip()
    elif title.startswith("Added chat from link: "):
        text = title[len("Added chat from link: "):].strip()
        if text:
            prompt = text

    if not prompt:
        return None

    # 过滤敏感占位符
    if prompt.strip().lower() in SENSITIVE_PATTERNS:
        return None

    response = ""
    items = record.get("safeHtmlItem", [])
    if items:
        html = items[0].get("html", "")
        response = strip_html(html)

    return {
        "prompt": prompt,
        "response": response,
        "time": datetime.fromisoformat(record["time"].replace("Z", "+00:00")),
    }


def filter_and_sort(records: list[dict]) -> list[dict]:
    filtered = []
    for r in records:
        parsed = extract_record(r)
        if parsed:
            filtered.append(parsed)
    filtered.sort(key=lambda x: x["time"])
    return filtered


def build_session(session_id: int, turns: list[dict]) -> dict:
    """
    将一组 turn 构建为领域分离的 Session 对象。
    turns: [{"prompt": ..., "response": ..., "time": datetime}, ...]
    """
    start_time = turns[0]["time"]
    end_time = turns[-1]["time"]
    duration_minutes = round((end_time - start_time).total_seconds() / 60, 1)

    # embedding_turns: 保留每个 turn 的独立 prompt（供语义拆分用）
    embedding_turns = [t["prompt"] for t in turns]
    # embedding_target: 用换行拼接（保留 turn 边界可读性）
    embedding_target = "\n".join(embedding_turns)

    # 构建 dialogues 列表
    dialogues = []
    for i, t in enumerate(turns, start=1):
        dialogues.append({
            "turn": i,
            "timestamp": t["time"].strftime(TIME_FMT),
            "user_prompt": t["prompt"],
            "ai_response": t["response"],
        })

    return {
        "session_id": f"sess_{int(start_time.timestamp())}",
        "metadata": {
            "start_time": start_time.strftime(TIME_FMT),
            "end_time": end_time.strftime(TIME_FMT),
            "duration_minutes": duration_minutes,
            "turn_count": len(turns),
        },
        "engine_data": {
            "embedding_turns": embedding_turns,
            "embedding_target": embedding_target,
            "umap_coords": None,
            "cluster_id": None,
        },
        "ui_data": {
            "dialogues": dialogues,
        },
    }


def sessionize(records: list[dict], gap_threshold: float = SESSION_GAP_THRESHOLD) -> list[dict]:
    if not records:
        return []

    sessions = []
    current_turns = [records[0]]

    for i in range(1, len(records)):
        gap = (records[i]["time"] - records[i - 1]["time"]).total_seconds()
        if gap > gap_threshold:
            sessions.append(build_session(len(sessions), current_turns))
            current_turns = [records[i]]
        else:
            current_turns.append(records[i])

    sessions.append(build_session(len(sessions), current_turns))
    return sessions


def run():
    print("[Phase 1] Loading raw data...")
    raw = load_raw_data()
    print(f"  Raw records: {len(raw)}")

    print("[Phase 1] Filtering and sorting...")
    cleaned = filter_and_sort(raw)
    print(f"  Valid prompt records: {len(cleaned)}")

    print(f"[Phase 1] Sessionizing with {SESSION_GAP_THRESHOLD / 60:.0f} min gap threshold...")
    sessions = sessionize(cleaned)
    # 过滤掉 embedding_target 为空的 session（全部 turn 都是敏感占位符的情况）
    before = len(sessions)
    sessions = [s for s in sessions if s["engine_data"]["embedding_target"].strip()]
    print(f"  Sessions created: {len(sessions)} (filtered {before - len(sessions)} empty)")

    turn_counts = [s["metadata"]["turn_count"] for s in sessions]
    print(f"  Turns per session — min: {min(turn_counts)}, "
          f"max: {max(turn_counts)}, "
          f"avg: {sum(turn_counts)/len(turn_counts):.1f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)
    print(f"[Phase 1] Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
