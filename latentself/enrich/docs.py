from pathlib import Path
import subprocess
from xml.etree import ElementTree
from zipfile import ZipFile

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract(path, name):
    if Path(name).suffix.lower() == ".pdf":
        text = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"],
                              check=True, capture_output=True).stdout.decode()
    else:
        with ZipFile(path) as archive:
            doc = ElementTree.fromstring(archive.read("word/document.xml"))
        text = "\n".join("".join(e.text or "" for e in p.iter(WORD + "t"))
                         for p in doc.iter(WORD + "p"))
    return text[:20000], max(0, len(text) - 20000)
