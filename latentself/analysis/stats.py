"""Calendar summaries with historical time zones and explicit group membership."""
from collections import Counter,defaultdict
import json
import os
from zoneinfo import ZoneInfo

from ..embed import DIALOGUE
from ..units import instant


def periods():
    values = json.loads(os.environ.get("LATENTSELF_TZ_PERIODS","[]"))
    return sorted((instant(v["from"]),ZoneInfo(v["zone"])) for v in values)


def local(ts,zones):
    stamp = instant(ts)
    zone = next((z for start,z in reversed(zones) if start<=stamp),ZoneInfo("UTC"))
    return stamp.astimezone(zone)


def overview(con):
    zones = periods()
    groups = defaultdict(Counter)
    for ts,prompt in con.execute(f"SELECT ts,prompt FROM turn WHERE kind IN {DIALOGUE}"):
        if ts:
            date = local(ts,zones)
            groups["month"][date.strftime("%Y-%m")] += 1
            groups["weekday"][str(date.isoweekday())] += 1
            groups["hour"][str(date.hour)] += 1
        else:
            groups["date_quality"]["unknown"] += 1
        length = next((n for n in (50,200,800,2000,6000) if len(prompt)<=n),None)
        groups["prompt_length"][f"≤{length}" if length else ">6000"] += 1
        zh = sum("\u3400"<=c<="\u9fff" for c in prompt)
        en = sum(c.isascii() and c.isalpha() for c in prompt)
        lang = "zh" if zh>en else "en" if en>zh else "mixed/other"
        groups["estimated_language"][lang] += 1
    for n,c in con.execute("SELECT n_turns,count(*) FROM conversation GROUP BY n_turns"):
        groups["conversation_turns"][str(n)] = c
    for start,end in con.execute("SELECT start_ts,end_ts FROM episode"):
        minutes = (instant(end)-instant(start)).total_seconds()/60
        bucket = next((n for n in (0,5,15,30,60,120) if minutes<=n),None)
        groups["episode_minutes"][f"≤{bucket}" if bucket is not None else ">120"] += 1
    for mime,c in con.execute("SELECT mime,count(*) FROM attachment GROUP BY mime"):
        groups["attachment_type"][mime or "unknown"] = c
    return dict(histograms={k:dict(v) for k,v in groups.items()},timezone_periods=len(zones),unconfigured_timezone="UTC")


def split_groups(con,preset):
    zones = periods()
    a,b = set(),set()
    split = os.environ.get("LATENTSELF_LIFE_DATE")
    bounds = preset.split("..") if ".." in preset else None
    if preset=="life" and not split:
        raise ValueError("set LATENTSELF_LIFE_DATE to the personal comparison date")
    if preset not in ("life","night","images","long") and (not bounds or len(bounds)!=2):
        raise ValueError("use life, night, images, long, or two comma-separated intervals joined by '..'")
    intervals = [s.split(",") for s in bounds] if bounds else []
    if intervals and (any(len(r)!=2 for r in intervals) or intervals[0][1]>=intervals[1][0]):
        raise ValueError("provide two disjoint inclusive date ranges as start,end..start,end")
    rows = con.execute(f"""SELECT t.id,t.ts,
        EXISTS(SELECT 1 FROM attachment i WHERE i.turn_id=t.id AND (i.mime LIKE 'image/%' OR i.origin='live_frame')),
        EXISTS(SELECT 1 FROM attachment i WHERE i.turn_id=t.id),
        (SELECT max(c.n_turns) FROM turn_conv tc JOIN conversation c ON c.id=tc.conv_id WHERE tc.turn_id=t.id)
        FROM turn t WHERE t.kind IN {DIALOGUE}""")
    for tid,ts,image,attachment,length in rows:
        if not ts and (intervals or preset in ("life","night")):
            continue
        date = local(ts,zones) if ts else None
        if intervals:
            for (start,end),dest in zip(intervals,(a,b)):
                if start<=date.date().isoformat()<=end:
                    dest.add(tid)
        elif preset=="life":
            (a if date.date().isoformat()<split else b).add(tid)
        elif preset=="night":
            (a if date.hour<5 else b).add(tid)
        elif preset=="images":
            if image or not attachment:
                (a if image else b).add(tid)
        elif length is not None:
            (a if length>=10 else b).add(tid)
    return a,b


def timeline(con,level=1.):
    rows = con.execute("""SELECT t.id,t.name,t.n_recur,t.active_days,t.first_ts,t.last_ts,
        substr(u.ts,1,7) AS month,count(DISTINCT tc.turn_id) AS turns FROM topic t
        JOIN topic_member m ON m.topic_id=t.id JOIN turn_conv tc ON tc.conv_id=m.conv_id
        JOIN turn u ON u.id=tc.turn_id WHERE t.resolution=? GROUP BY t.id,month
        ORDER BY t.n_recur DESC,t.active_days DESC,t.id,month""",(level,))
    result = {}
    for row in rows:
        topic = result.setdefault(row["id"],dict(id=row["id"],name=row["name"],n_recur=row["n_recur"],
            active_days=row["active_days"],first_ts=row["first_ts"],last_ts=row["last_ts"],months={}))
        topic["months"][row["month"] or "unknown"] = row["turns"]
    for topic in result.values():
        topic["duration_days"] = ((instant(topic["last_ts"])-instant(topic["first_ts"])).days
                                  if topic["first_ts"] and topic["last_ts"] else None)
    return list(result.values())
