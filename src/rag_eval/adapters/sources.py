import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class SourceError(ValueError):
    pass


def load_sources(directory: Path, names: set[str]) -> dict[str, str]:
    """Load the text of each named source document found in `directory`.

    `name.txt` or `name.md` next to the original wins, so a PDF is checked
    against its extracted text (`pdftotext name.pdf name.pdf.txt`). DOCX files
    are read directly. Names with no readable file are left out.
    """
    texts = {}
    for name in sorted(names):
        for candidate in (directory / f"{name}.txt", directory / f"{name}.md", directory / name):
            if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md", ".docx"}:
                texts[name] = read_text(candidate)
                break
    return texts


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _docx_text(path)
    return path.read_text(encoding="utf-8")


def _docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise SourceError(f"{path}: not a readable DOCX file ({e})") from e
    paragraphs = ["".join(t.text or "" for t in p.iter(f"{_W}t")) for p in root.iter(f"{_W}p")]
    return re.sub(r"\n{2,}", "\n", "\n".join(paragraphs)).strip()
