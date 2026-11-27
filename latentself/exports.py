"""Write private Markdown exports while preserving edits made outside LatentSelf."""
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def export_markdown(con,path,body):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data = (body.rstrip()+"\n").encode()
    digest = sha256(data).hexdigest()
    key = "export:"+str(path.absolute())
    with con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT v FROM meta WHERE k=?",(key,)).fetchone()
        current = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if current==digest and not row:
            return path
        if current!=digest and (path.is_symlink() or current and (not row or row[0]!=current)):
            # The canonical document belongs to the owner. Keep this revision alongside it.
            path = path.with_stem(path.stem+"-"+digest[:12])
            key = "export:"+str(path.absolute())
            current = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if path.is_symlink() or current and current!=digest:
                raise FileExistsError(path)
        if current!=digest:
            with TemporaryDirectory(dir=path.parent) as tmp:
                staged = Path(tmp)/"document"
                with staged.open("xb") as f:
                    os.fchmod(f.fileno(),0o600)
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                if current is None:
                    os.link(staged,path)
                else:
                    os.replace(staged,path)
        con.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",(key,digest))
    return path
