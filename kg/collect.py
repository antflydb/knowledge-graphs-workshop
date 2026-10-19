"""Access-checked collection of fixed inventories into Markdown and provenance JSONL."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import httpx

from kg.dataset_inputs import _data, collect_command, raw_path, read_jsonl, read_manifest
from kg.wikipedia import Fetcher, parse_article


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class WebFetcher:
    """Paced requests; robots and redirects checked independently for every origin."""
    def __init__(self, user_agent, interval):
        self.user_agent, self.interval = user_agent, interval
        self.client = httpx.Client(headers={"User-Agent": user_agent}, timeout=90, follow_redirects=False)
        self.robots, self.last = {}, {}

    def _get(self, url):
        origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
        time.sleep(max(0, self.last.get(origin, 0) + self.interval - time.monotonic()))
        try:
            return self.client.get(url)
        finally:
            self.last[origin] = time.monotonic()

    def _allowed(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username:
            raise ValueError(f"Refusing non-HTTPS or credential-bearing URL: {url}")
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            response = self._get(origin + "/robots.txt")
            # Only missing robots permits unrestricted access. Blocks and server
            # failures are explicit failures, never mistaken for an empty policy.
            robot = RobotFileParser()
            if response.status_code in (404, 410):
                robot.parse([])
            elif response.status_code == 200:
                if "text/html" in response.headers.get("content-type", "").lower() and "<html" in response.text.lower():
                    raise ValueError(f"robots.txt returned an HTML page: {origin}")
                robot.parse(response.text.splitlines())
            elif response.is_redirect:
                redirected = urljoin(str(response.url), response.headers["location"])
                if urlsplit(redirected).netloc != parsed.netloc:
                    raise ValueError(f"robots.txt redirects off origin: {origin} -> {redirected}")
                response = self._get(redirected)
                response.raise_for_status()
                if response.status_code != 200 or "<html" in response.text.lower():
                    raise ValueError(f"Unusable robots.txt: {redirected}")
                robot.parse(response.text.splitlines())
            else:
                response.raise_for_status()
                raise ValueError(f"Unusable robots.txt HTTP {response.status_code}: {origin}")
            self.robots[origin] = robot
        if not self.robots[origin].can_fetch(self.user_agent, url):
            raise ValueError(f"robots.txt disallows {url}")

    def fetch(self, url):
        for _ in range(6):
            self._allowed(url)
            response = self._get(url)
            if response.is_redirect:
                url = urljoin(str(response.url), response.headers["location"])
                continue
            response.raise_for_status()
            raw = response.content
            return raw, {"download_url": str(response.url), "source_sha256": sha(raw),
                         "fetched_at": datetime.now(timezone.utc).isoformat(),
                         "source_media_type": response.headers.get("content-type", "")}
        raise ValueError(f"Too many redirects: {url}")


def html_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup.select("script,style,noscript,template,nav,form,iframe,svg,[role=navigation],[hidden],.cookie-banner,.cookie-notice,#cookie-banner,#cookie-notice,.cookie-consent,.Header-nav,.Header-nav-inner,.Mobile-overlay-nav,.Mobile-overlay-nav-item,.Mobile-overlay-folder,.Cart,.sqs-custom-cart"):
        node.decompose()
    body = soup.body or soup
    # Older sites use link-only paragraphs instead of semantic nav elements.
    # Remove the navigation paragraph, never its parent containing hours/address.
    for node in body.find_all("p"):
        links = node.find_all("a", href=True)
        if len({a["href"] for a in links if not a["href"].startswith("#")}) >= 3 and not any(
            s.strip() for s in node.strings if s.find_parent("a") is None
        ):
            node.decompose()
    for node in body.find_all(["br", "p", "div", "section", "article", "header", "footer", "li", "h1", "h2", "h3", "h4", "tr"]):
        if node.name == "br":
            node.replace_with("\n")
        else:
            node.insert_before("\n")
            node.insert_after("\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in body.get_text().splitlines()]
    text = "\n".join(line for line in lines if line).strip() + "\n"
    if len(text.strip()) < 40:
        raise ValueError("HTML has no usable visible prose/menu text (possibly an image or JavaScript shell)")
    return text


def pdf_text(raw):
    if not raw.startswith(b"%PDF-"):
        raise ValueError("Menu source is not a PDF (%PDF- signature missing)")
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError
    try:
        reader = PdfReader(BytesIO(raw))
        pages, omitted, coverage = [], [], []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            coverage.append({"page": number, "text_characters": len(text.strip())})
            if text.strip():
                pages.append(text)
            else:
                omitted.append(number)
    except PyPdfError as error:
        raise ValueError(f"PDF text extraction failed: {error}") from error
    if not pages:
        raise ValueError("Menu PDF has no extractable text; OCR/image substitution is not permitted")
    return "\n\n".join(pages), {"pdf_pages": len(reader.pages), "pages_without_text": omitted,
                              "page_text_coverage": coverage,
                              "coverage_note": "Extracted text only; a page with text may still contain unextracted image content. Character counts do not imply complete menu coverage."}


def transcript_text(raw, window, refresh):
    if not refresh and sha(raw) != window["source_sha256"]:
        raise ValueError("Transcript download differs from the approved snapshot; review and use collect --refresh explicitly")
    text = raw.decode(window["encoding"]).replace("\r\n", "\n").replace("\r", "\n")
    def boundary(met):
        found = list(re.finditer(r"(?m)^[ \t]*" + re.escape(met) + r"[ \t]+[A-Z]+\b", text))
        if len(found) != 1:
            raise ValueError(f"Expected exactly one transcript boundary {met}; found {len(found)}")
        return found[0].start()
    start, end, last = boundary(window["start"]), boundary(window["next"]), boundary(window["end"])
    if not start < last < end:
        raise ValueError("Transcript window boundaries are not in source order")
    selected = text[start:end]
    if not refresh and sha(selected.encode("utf-8")) != window["sha256"]:
        raise ValueError("Normalized transcript excerpt differs from the approved snapshot; review collect --refresh")
    return selected, {"encoding": window["encoding"], "normalization": "Decoded original bytes; CRLF/CR normalized to LF; UTF-8 output; no text corrections",
                      "source_start": start, "source_end": end, "offset_unit": "decoded_lf_normalized_python_characters",
                      "met_start": window["start"], "met_end": window["end"],
                      "boundary_next": window["next"], "role_legend": {"CDR": "Commander James A. Lovell", "CMP": "Command module pilot John L. Swigert", "LMP": "Lunar module pilot Fred W. Haise", "CC": "CapCom role, not one fixed person", "SC": "Unidentified spacecraft speaker", "MS": "Multiple speakers"}}


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def collect(cfg, refresh=False):
    cfg = _data(cfg)
    settings = cfg["collection"]
    if settings["kind"] == "bundled":
        # Checked-in sources: nothing to fetch, only verify the snapshot.
        return {"bundled": True, "sources": len(read_manifest(cfg))}
    manifest = Path(cfg["corpus"])
    existing = read_jsonl(manifest) if manifest.exists() else []
    if len({r.get("id") for r in existing}) != len(existing):
        raise ValueError("Duplicate existing manifest source IDs; repair metadata before collection")
    inventory_ids = {s["id"] for s in settings["sources"]}
    if any(not isinstance(r.get("id"), str) or r["id"] not in inventory_ids for r in existing):
        raise ValueError("Existing manifest contains missing or unconfigured source IDs; repair it before collection")
    rows = {r["id"]: r for r in existing}
    failures, fetched = [], 0
    fetcher = None
    try:
        for source in settings["sources"]:
            path = raw_path(cfg, source["path"])
            previous = rows.get(source["id"])
            if path.exists() and previous is None and not refresh:
                failures.append({"id": source["id"], "url": source["url"],
                                 "error": f"Unregistered existing raw source {path}; preserve/review it, then explicitly run {collect_command(cfg)} --refresh"})
                continue
            if path.exists() and previous and not refresh:
                if previous.get("path") != source["path"] or sha(path.read_bytes()) != previous.get("sha256"):
                    failures.append({"id": source["id"], "url": source["url"], "error": f"Raw snapshot mismatch at {path}; restore it or run {collect_command(cfg)} --refresh, then rebuild"})
                continue
            try:
                if fetcher is None:
                    if settings["kind"] == "wikipedia":
                        cache = Path(cfg["output"]) / "collection-cache"
                        if refresh:
                            cache = cache / datetime.now(timezone.utc).strftime("refresh-%Y%m%dT%H%M%S%f")
                        fetcher = Fetcher(cache, settings["user_agent"], settings["request_interval_seconds"], False)
                    else:
                        fetcher = WebFetcher(settings["user_agent"], settings["request_interval_seconds"])
                url = source.get("download_url", source["url"])
                if settings["kind"] == "wikipedia":
                    html, meta = fetcher.fetch(url)
                    parsed, _, _ = parse_article(html, {s.casefold() for s in settings["exclude_sections"]})
                    if parsed["page_id"] != source["page_id"] or (not refresh and parsed["revision_id"] != source["revision_id"]):
                        raise ValueError("Wikipedia page/revision differs from fixed inventory; explicit collect --refresh required")
                    text = parsed.pop("text")
                    parsed.pop("text_sha256", None)
                    row = {**parsed, "source_sha256": meta["sha256"], "fetched_at": meta["fetched_at"], "source_kind": "encyclopedia_prose",
                           "attribution": "Wikipedia contributors; " + source["url"], "redistribution": "allowed",
                           "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={parsed['revision_id']}"}
                else:
                    raw, meta = fetcher.fetch(url)
                    extra = {}
                    if source["format"] == "pdf":
                        text, extra = pdf_text(raw)
                    elif source["format"] == "html":
                        if "text/html" not in meta["source_media_type"].lower() and b"<html" not in raw[:1000].lower():
                            raise ValueError("HTML source returned a non-HTML response")
                        text = html_text(raw)
                    elif source["format"] == "transcript":
                        if meta["source_media_type"].split(";", 1)[0].strip().lower() != "text/plain":
                            raise ValueError(f"Transcript source must return text/plain, got {meta['source_media_type']!r}")
                        text, extra = transcript_text(raw, settings["transcript_window"], refresh)
                    else:
                        if meta["source_media_type"].split(";", 1)[0].strip().lower() != "text/plain":
                            raise ValueError(f"Publisher report source must return text/plain, got {meta['source_media_type']!r}")
                        if not refresh and source.get("source_sha256") and sha(raw) != source["source_sha256"]:
                            raise ValueError("Downloaded text differs from approved snapshot; use collect --refresh explicitly")
                        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                    if settings["kind"] == "apollo13":
                        kind = "mission_transcript" if source["format"] == "transcript" else "review_board"
                        attribution = "NASA; " + ("plain-text formatting by Heiko Kueffen; https://apollojournals.org/alsj/a13/a13trans.html" if kind == "mission_transcript" else "Apollo 13 Review Board; NASA NTRS publisher-supplied fulltext")
                        row = {**meta, **extra, "source_kind": kind, "license": "Work of the US Government; public use permitted", "attribution": attribution, "redistribution": "allowed"}
                    else:
                        row = {**meta, **extra, "source_kind": "menu_about" if len(source["roles"]) > 1 else source["roles"][0],
                               "restaurant_id": source["restaurant_id"], "roles": source["roles"], "license": None,
                               "attribution": source["title"] + "; " + source["url"], "redistribution": "permission_not_verified",
                               "modifications": "Extracted available PDF page text in order; no OCR" if source["format"] == "pdf" else "Visible HTML text; scripts, style, navigation, forms and cookie scaffolding removed; branding/contact/hours retained"}
                if not text.strip():
                    raise ValueError("Source yielded empty text")
                encoded = text.encode("utf-8")
                expected = previous.get("sha256") if previous else source.get("sha256")
                if expected and sha(encoded) != expected and not refresh:
                    raise ValueError(f"Collected Markdown differs from expected hash; run {collect_command(cfg)} --refresh after reviewing the source")
                row.update(id=source["id"], path=source["path"], title=source["title"], url=source["url"], media_type="text/markdown", sha256=sha(encoded))
                _atomic_text(path, text)
                rows[source["id"]] = row
                fetched += 1
                # Persist each successful source, preserving previous successful rows
                # if a later source fails or collection is interrupted.
                ordered = [rows[s["id"]] for s in settings["sources"] if s["id"] in rows]
                _atomic_text(manifest, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
            except (ValueError, UnicodeError, httpx.HTTPError, RuntimeError, OSError) as error:
                failures.append({"id": source["id"], "url": source.get("download_url", source["url"]), "error": str(error)})
    finally:
        if fetcher is not None:
            fetcher.client.close()
    if not failures:
        try:
            read_manifest(cfg)
        except ValueError as error:
            failures.append({"id": "manifest", "url": str(manifest), "error": str(error)})
    report = {"dataset": cfg["dataset"], "sources": len(settings["sources"]), "fetched": fetched, "refresh": refresh,
              "failures": failures, "complete": not failures, "recorded_at": datetime.now(timezone.utc).isoformat()}
    _atomic_text(Path(cfg["output"]) / "collection.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if failures:
        details = "; ".join(f"{f['url']}: {f['error']}" for f in failures)
        raise ValueError(f"Collection incomplete; successful snapshots preserved. {details}. Report: {Path(cfg['output']) / 'collection.json'}")
    return report
