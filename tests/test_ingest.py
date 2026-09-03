# tests/test_ingest.py
from voiceagent.deploy import ingest

def _stub_fetcher_factory(pages):
    def fetch(url):
        if url not in pages:
            raise OSError(f"blocked or missing: {url}")
        return pages[url], url
    return fetch

def test_caps_and_same_origin():
    pages = {"https://acme.test/": '<a href="/a">a</a><a href="https://evil.test/x">x</a><p>Hello</p>',
             "https://acme.test/a": "<p>Page A</p>"}
    chunks = ingest.fetch_site("https://acme.test/", fetcher=_stub_fetcher_factory(pages))
    sources = [c["source"] for c in chunks]
    assert sources and all(s.startswith("https://acme.test/") for s in sources)
    assert not any("evil.test" in s for s in sources)

def test_allowlist_capped_at_3():
    import pytest
    with pytest.raises(ValueError, match="allowlist"):
        ingest.fetch_site("https://acme.test/",
                          allowlist=["https://a.test/", "https://b.test/",
                                     "https://c.test/", "https://d.test/"],
                          fetcher=_stub_fetcher_factory({}))

def test_max_pages_cap():
    pages = {f"https://acme.test/p{i}": "<p>t</p>" for i in range(200)}
    pages["https://acme.test/"] = "".join(
        f'<a href="/p{i}">x</a>' for i in range(200)) + "<p>home</p>"
    chunks = ingest.fetch_site("https://acme.test/", fetcher=_stub_fetcher_factory(pages))
    assert len(chunks) <= ingest.MAX_PAGES

def test_owner_paste_ranks_first():
    pasted = [ingest.ingest_owner_paste("Refund window is 30 days", label="policy-doc")]
    crawled = [{"text": "old text", "source": "https://acme.test/", "crawled_at": "2020-01-01T00:00:00Z"}]
    ranked = ingest.rank_chunks(pasted, crawled)
    assert ranked[0]["source"] == "owner_paste:policy-doc"

def test_stub_fetcher_skips_robots_entirely(monkeypatch):
    import urllib.request
    def _boom(*a, **k):
        raise AssertionError("network used despite stub fetcher")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    pages = {"https://acme.test/": "<p>Hello</p>",
             "https://acme.test/private": "<p>Secret</p>"}
    chunks = ingest.fetch_site("https://acme.test/", fetcher=_stub_fetcher_factory(pages))
    assert chunks and chunks[0]["source"] == "https://acme.test/"
    assert not any(c["source"].startswith("gap:robots") for c in chunks)
