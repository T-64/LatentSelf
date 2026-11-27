"""Cross-check TSPDT's published identifiers without transferring the workbook to the Mac."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unicodedata


def signature(title,director,year):
    def fold(value):
        return "".join(c for c in unicodedata.normalize("NFKD",str(value or "")).casefold() if c.isalnum())
    # The published table includes multi-year productions. The catalog uses their first year.
    start = re.fullmatch(r"(\d{4})(?:\.0|\s*[-–/]\s*\d{2,4})?",str(year).strip())
    if not start:
        raise ValueError("unrecognized release-year format")
    return sha256("\n".join([fold(title),fold(director),start[1]]).encode()).hexdigest()


def extract(path,output,year=2026):
    if sys.platform!="linux":
        raise ValueError("read the public workbook on the inference server")
    import xlrd
    book = xlrd.open_workbook(path,on_demand=True)
    try:
        sheet = book.sheet_by_name("StartingList")
        header = [str(v) for v in sheet.row_values(0)]
        pos,title,creator,release,ext = [header.index(v) for v in [str(float(year)),"Title","Director(s)","Year","IMDb"]]
        result = {}
        for i in range(1,sheet.nrows):
            rank = sheet.cell_value(i,pos)
            try:
                rank = int(float(rank))
            except (TypeError,ValueError):
                continue
            if not 1<=rank<=1000:
                continue
            link = sheet.hyperlink_map.get((i,ext))
            match = re.search(r"/title/(tt\d{5,12})(?:/|[?#]|$)",str(link.url_or_path if link else ""))
            if not match:
                continue
            try:
                digest = signature(sheet.cell_value(i,title),sheet.cell_value(i,creator),sheet.cell_value(i,release))
            except ValueError:
                continue
            if rank in result:
                raise ValueError("ambiguous rank in the published workbook")
            result[rank] = dict(signature=digest,imdb=match[1])
    finally:
        book.release_resources()
    data = json.dumps(dict(edition=year,records=result),sort_keys=True).encode()
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if output.read_bytes()!=data:
            raise FileExistsError("the existing public index describes a different workbook")
    else:
        with TemporaryDirectory(dir=output.parent) as tmp:
            staged = Path(tmp)/"index"
            staged.write_bytes(data)
            os.link(staged,output)
    return dict(edition=year,public_ids=len(result),workbook_sha256=sha256(Path(path).read_bytes()).hexdigest())


def server_index():
    raw = subprocess.check_output(["ssh","-o","BatchMode=yes","server","cat \"$HOME/.cache/latentself/tspdt-2026-ids.json\""])
    data = json.loads(raw)
    if data["edition"]!=2026 or len(data["records"])>1000:
        raise ValueError("unexpected public index edition or size")
    return data["records"]


def match_public_ids(rows,records):
    result = {}
    for row in rows:
        record = records.get(str(row["rank"]))
        if not record or not re.fullmatch(r"tt\d{5,12}",record.get("imdb","")):
            continue
        if any(signature(title,row["creator"],row["year"])==record["signature"]
               for title in (row.get("rawTitle"),row["title"]) if title):
            result[row["canonicalKey"]] = record["imdb"]
    return result


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    print(json.dumps(extract(args.workbook,args.output)))
