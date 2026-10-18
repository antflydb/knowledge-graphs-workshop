"""Wikipedia prose parser and access-checked Fetcher for fixed-list collection."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import time
from urllib.parse import unquote, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import httpx

HOST = "https://en.wikipedia.org"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def title_from_link(href):
    url = urlsplit(urljoin(HOST, href))
    if url.netloc != "en.wikipedia.org" or not url.path.startswith("/wiki/") or url.query:
        return None
    title = unquote(url.path[6:]).replace("_", " ")
    if not title or (":" in title and not title.startswith("Category:")):
        return None
    return title


def normalize(text):
    return " ".join(text.split())


def parse_article(html, excluded):
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("#mw-content-text .mw-parser-output") or soup.select_one(".mw-parser-output")
    if body is None:
        raise ValueError("Missing article body")
    canonical = soup.select_one('link[rel="canonical"]')
    title = soup.select_one("#firstHeading")
    revision = re.search(r'"wgRevisionId"\s*:\s*(\d+)', html)
    page_id = re.search(r'"wgArticleId"\s*:\s*(\d+)', html)
    footer = soup.select_one("#footer-info-copyright")
    if not canonical or not title or not revision or not page_id or not footer:
        raise ValueError("Missing canonical URL, page/revision identity or license notice")
    if not footer.find("a", href=re.compile(r"(creativecommons.org/licenses/by-sa/4.0|Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License)")):
        raise ValueError("Unrecognized text license; review before redistributing")
    discovery = set()
    for hatnote in body.select(".hatnote"):
        if not re.match(r"(Main article|Further information)", hatnote.get_text(" ", strip=True)):
            continue
        for a in hatnote.select("a[href]"):
            linked = title_from_link(a["href"])
            if linked and not linked.startswith("Category:"):
                discovery.add(linked)
    headings = {normalize(h.get_text(" ", strip=True)).removesuffix("[edit]").strip().casefold()
                for h in body.select("h2,h3,h4,h5,h6")}
    heading_links = set()
    for a in body.select("a[href]"):
        linked = title_from_link(a["href"])
        if linked and normalize(a.get_text(" ", strip=True)).casefold() in headings and (a.find_parent(["h2", "h3", "h4", "h5", "h6"]) or linked.casefold() in headings):
            heading_links.add(linked)
    # Tables/infoboxes and editorial scaffolding are deliberately not evidence.
    for item in body.select("table,figure,.thumb,.hatnote,.navbox,.reflist,.reference,.mw-editsection,.metadata,.toc,style,script,blockquote"):
        item.decompose()
    path, parts, sections, skip_depth = [], [], [], None
    cursor = 0
    for item in body.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if item.name in ("p", "li") and item.find_parent(["li", "p"]):
            continue
        if item.name.startswith("h"):
            depth = int(item.name[1])
            heading = normalize(item.get_text(" ", strip=True))
            path = [(d, h) for d, h in path if d < depth] + [(depth, heading)]
            if skip_depth is not None and depth <= skip_depth:
                skip_depth = None
            if heading.casefold() in excluded:
                skip_depth = depth
            continue
        if skip_depth is not None:
            continue
        text = normalize(item.get_text(" ", strip=True))
        if not text:
            continue
        # Preserve headings as provenance only; omit them from extraction text.
        if parts:
            cursor += 2
        start = cursor
        parts.append(text)
        cursor += len(text)
        sections.append({"start_char": start, "end_char": cursor, "headings": [h for _, h in path]})
    text = "\n\n".join(parts)
    if not text:
        raise ValueError("No article prose")
    return {"title": normalize(title.get_text(" ", strip=True)), "url": canonical["href"],
            "page_id": int(page_id[1]), "revision_id": int(revision[1]), "text": text,
            "sections": sections, "text_sha256": sha(text.encode()),
            "license": LICENSE, "license_notice": normalize(footer.get_text(" ", strip=True)),
            "canon_scope": "Mixed book/adaptation coverage; scope must be established from passage text.",
            "modifications": "Prose normalized; headings retained separately; tables, infoboxes, images, block quotations and configured editorial sections omitted."}, discovery, heading_links


class Fetcher:
    def __init__(self, root, user_agent, interval, cached):
        self.root, self.interval, self.cached = root, interval, cached
        self.last = 0.0
        self.client = httpx.Client(headers={"User-Agent": user_agent}, timeout=60, follow_redirects=False)
        self.user_agent = user_agent
        root.mkdir(parents=True, exist_ok=True)
        self.robot = RobotFileParser()
        raw, _ = self.fetch(HOST + "/robots.txt", check=False)
        self.robot.parse(raw.splitlines())

    def fetch(self, url, check=True):
        for _ in range(6):
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.netloc != "en.wikipedia.org":
                raise ValueError(f"Wikipedia Fetcher refuses off-site URL: {url}")
            if check and not self.robot.can_fetch(self.user_agent, url):
                raise ValueError(f"robots.txt disallows {url}")
            key = sha(url.encode())
            raw_path, meta_path = self.root / (key + ".html"), self.root / (key + ".json")
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                if "redirect" in metadata:
                    url = metadata["redirect"]
                    continue
            if raw_path.exists() and meta_path.exists():
                raw, meta = raw_path.read_bytes(), json.loads(meta_path.read_text())
                if sha(raw) != meta["sha256"]:
                    raise ValueError(f"Cache hash mismatch: {url}")
                return raw.decode("utf-8"), meta
            if self.cached:
                raise ValueError(f"Missing cached response: {url}")
            time.sleep(max(0, self.last + self.interval - time.monotonic()))
            response = self.client.get(url)
            self.last = time.monotonic()
            if response.status_code in (301, 302, 303, 307, 308):
                redirected = urljoin(url, response.headers["location"])
                if urlsplit(redirected).netloc != "en.wikipedia.org":
                    raise ValueError("Refusing off-site redirect")
                meta_path.write_text(json.dumps({"redirect": redirected}) + "\n")
                url = redirected
                continue
            # Never retry access blocks or throttles in a tight loop. A later run
            # resumes from cache after the operator reviews the response.
            if response.status_code in (429, 503):
                raise RuntimeError(f"Collection paused: HTTP {response.status_code}; Retry-After={response.headers.get('retry-after')}")
            response.raise_for_status()
            raw = response.content
            meta = {"url": str(response.url), "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": sha(raw), "content_type": response.headers.get("content-type"),
                    "etag": response.headers.get("etag"), "last_modified": response.headers.get("last-modified")}
            raw_path.write_bytes(raw)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            return raw.decode("utf-8"), meta
        raise ValueError("Too many redirects")


