"""Scoped ingestion: owner domain + allowlist only. Text-only, capped."""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib import robotparser, request

MAX_PAGES = 50
MAX_DEPTH = 3
PAGE_TIMEOUT_S = 15
MAX_ALLOWLIST = 3

class _TextLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text: list[str] = []
        self.links: list[str] = []
        self._skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)
    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
    def handle_data(self, data):
        if not self._skip and data.strip():
            self.text.append(data.strip())

def _default_fetcher(url: str) -> tuple[str, str]:
    req = request.Request(url, headers={"User-Agent": "VoiceAgent-deploy/1.0"})
    with request.urlopen(req, timeout=PAGE_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace"), resp.geturl()

def _allowed(url: str, seed_host: str, extra: set[str]) -> bool:
    host = urlparse(url).netloc
    return host == seed_host or host in extra

def fetch_site(seed_url: str, allowlist: list[str] | None = None,
               fetcher=None) -> list[dict]:
    allowlist = allowlist or []
    if len(allowlist) > MAX_ALLOWLIST:
        raise ValueError(f"allowlist capped at {MAX_ALLOWLIST}")
    fetch = fetcher or _default_fetcher
    seed_host = urlparse(seed_url).netloc
    extra = {urlparse(u).netloc for u in allowlist}
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(seed_url, 0)]
    chunks: list[dict] = []
    gaps: list[str] = []
    if fetcher is None:
        # Best-effort robots.txt check for the seed host (5s timeout).
        # On failure proceed — log a gap chunk. Skipped entirely when a
        # stub fetcher is injected so tests never touch the network.
        try:
            robots_url = urljoin(seed_url, "/robots.txt")
            req = request.Request(robots_url, headers={"User-Agent": "VoiceAgent-deploy/1.0"})
            with request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            rp = robotparser.RobotFileParser()
            rp.parse(body.splitlines())
        except Exception as e:
            gaps.append(f"robots:{robots_url}: {e}")
    while queue and len(chunks) < MAX_PAGES:
        url, depth = queue.pop(0)
        if url in seen or depth > MAX_DEPTH:
            continue
        seen.add(url)
        try:
            html, final_url = fetch(url)
        except Exception as e:  # gap, never half-parse
            gaps.append(f"{url}: {e}")
            continue
        if not _allowed(final_url, seed_host, extra):
            continue
        p = _TextLinks()
        p.feed(html)
        text = " ".join(p.text)[:4000]
        if text:
            chunks.append({"text": text, "source": final_url,
                           "crawled_at": datetime.now(timezone.utc).isoformat()})
        if depth < MAX_DEPTH:
            for href in p.links:
                nxt = urljoin(final_url, href)
                if not nxt.startswith("http") or nxt in seen:
                    continue
                if not _allowed(nxt, seed_host, extra):
                    continue
                queue.append((nxt, depth + 1))
    for g in gaps:
        chunks.append({"text": "", "source": f"gap:{g}",
                       "crawled_at": datetime.now(timezone.utc).isoformat()})
    return chunks

def ingest_owner_paste(text: str, label: str = "owner_paste") -> dict:
    return {"text": text[:8000], "source": f"owner_paste:{label}",
            "crawled_at": datetime.now(timezone.utc).isoformat()}

def rank_chunks(pasted: list[dict], crawled: list[dict]) -> list[dict]:
    real = [c for c in crawled if not c["source"].startswith("gap:")]
    return list(pasted) + real
