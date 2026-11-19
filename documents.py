import unicodedata

from pypdf import PdfReader


def load_paper(path):
    """Return a paper's text without its reference list and anything after it."""
    text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    end = max(text.rfind("References"), text.rfind("REFERENCES"))
    text = unicodedata.normalize("NFKC", text[:end])
    text = text.replace("-\n", "")  # words split across lines
    return " ".join(text.split())


def load_papers(folder):
    return {path.stem: load_paper(path) for path in sorted(folder.glob("*.pdf"))}


def make_chunks(papers, tokenizer, size, overlap):
    """Split each paper into chunks of `size` LLM tokens. Chunks never span two papers."""
    chunks = []
    for name, text in papers.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        for number, start in enumerate(range(0, len(ids), size - overlap)):
            piece = ids[start:start + size]
            chunks.append({"id": f"{name}#{number}", "text": tokenizer.decode(piece), "tokens": len(piece)})
            if start + size >= len(ids):
                break
    return chunks
