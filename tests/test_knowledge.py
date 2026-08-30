import tempfile
from pathlib import Path
from voiceagent.knowledge import load_docs, build_index


def test_load_docs_parses_markdown_sections():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "faqs.md").write_text(
            "# Returns\nYou can return within 7 days.\n\n"
            "# Refunds\nRefunds take 5-7 business days.\n"
        )
        docs = load_docs(str(d))
        texts = [x["text"] for x in docs]
        assert any("7 days" in t for t in texts)
        assert any("5-7 business days" in t for t in texts)


def test_build_index_and_search_returns_relevant_doc():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "faqs.md").write_text(
            "# Returns\nYou can return any item within 7 days of delivery.\n\n"
            "# Refunds\nRefunds are processed within 5-7 business days.\n"
        )
        docs = load_docs(str(d))
        idx = build_index(docs)
        results = idx.search("how long do refunds take?", k=1)
        assert len(results) == 1
        assert "5-7 business days" in results[0]["text"]