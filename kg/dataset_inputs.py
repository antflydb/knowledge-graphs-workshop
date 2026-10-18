"""Strict Markdown manifests and exact, whole-word source-offset passages."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from kg.dataset_config import digest


def _data(cfg):
    return cfg.data if hasattr(cfg, "data") else cfg


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: malformed JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: metadata must be a JSON object")
            rows.append(row)
    return rows


def raw_path(cfg, relative):
    cfg = _data(cfg)
    root = Path(cfg["config_path"]).parent
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Manifest path must be dataset-relative: {relative!r}")
    path = (root / relative).resolve()
    raw_root = (root / "raw").resolve()
    if not raw_root.is_relative_to(root.resolve()) or not path.is_relative_to(raw_root) or path.suffix != ".md":
        raise ValueError(f"Manifest path must remain under raw/ with .md extension: {relative}")
    return path


def collect_command(cfg):
    cfg = _data(cfg)
    return f"python -m kg.dataset {cfg['config_path']} --track {cfg['track']} collect"


def read_manifest(cfg):
    """Return validated records with exact UTF-8 text, never persisting inline text."""
    cfg = _data(cfg)
    manifest = Path(cfg["corpus"])
    if not manifest.exists():
        raise ValueError(f"Missing metadata manifest {manifest}; run {collect_command(cfg)}")
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError(f"Empty metadata manifest {manifest}; run {collect_command(cfg)}")
    required = {"id", "path", "title", "url", "media_type", "sha256", "fetched_at", "license", "attribution", "redistribution", "source_kind"}
    ids, paths, result = set(), set(), []
    for row in rows:
        missing = required - set(row)
        if missing or "text" in row or "turns" in row:
            raise ValueError(f"Malformed manifest record {row.get('id')}: missing={sorted(missing)}; inline text/turns forbidden")
        for key in required - {"license"}:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Invalid manifest {key}: {row.get('id')}")
        if row["license"] is not None and not isinstance(row["license"], str):
            raise ValueError(f"Invalid license: {row['id']}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate source id: {row['id']}")
        ids.add(row["id"])
        path = raw_path(cfg, row["path"])
        if path in paths:
            raise ValueError(f"Duplicate source path: {row['path']}")
        paths.add(path)
        if row["media_type"] != "text/markdown" or row["redistribution"] not in {"allowed", "permission_not_verified"}:
            raise ValueError(f"Invalid media type/redistribution: {row['id']}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid raw sha256: {row['id']}")
        if "source_sha256" in row and (not isinstance(row["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])):
            raise ValueError(f"Invalid source_sha256: {row['id']}")
        try:
            fetched = datetime.fromisoformat(row["fetched_at"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid fetched_at: {row['id']}") from error
        if fetched.tzinfo is None:
            raise ValueError(f"fetched_at requires a timezone: {row['id']}")
        if not path.is_file():
            raise ValueError(f"Missing raw source {path}; run {collect_command(cfg)}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Raw hash mismatch: {path}; restore the snapshot or explicitly run {collect_command(cfg)} --refresh, then rebuild")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Raw Markdown must be UTF-8: {path}") from error
        if not text.strip():
            raise ValueError(f"Empty raw source: {path}")
        sections = row.get("sections", [])
        if not isinstance(sections, list):
            raise ValueError(f"sections must be a list: {row['id']}")
        previous_end = 0
        for section in sections:
            if not isinstance(section, dict) or not {"start_char", "end_char"} <= set(section):
                raise ValueError(f"Malformed section: {row['id']}")
            start, end = section["start_char"], section["end_char"]
            if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(text):
                raise ValueError(f"Invalid section offsets: {row['id']}")
            headings = section.get("headings", [])
            if not isinstance(headings, list) or any(not isinstance(h, str) for h in headings):
                raise ValueError(f"Invalid section headings: {row['id']}")
            previous_end = end
        result.append({**row, "text": text})
    root = Path(cfg["config_path"]).parent / "raw"
    extra = {p.resolve() for p in root.rglob("*") if p.is_file()} - paths
    if extra:
        raise ValueError(f"Raw files absent from manifest: {', '.join(str(p) for p in sorted(extra))}")
    inventory = {s["id"]: s["path"] for s in cfg["collection"]["sources"]}
    if {r["id"]: r["path"] for r in result} != inventory:
        raise ValueError(f"Manifest does not cover the fixed collection inventory; run {collect_command(cfg)}")
    return result


def windows(text, *, max_bytes, max_words, overlap_words):
    """Whole-word windows with Python character offsets into the unmodified text."""
    words = list(re.finditer(r"\S+", text))
    first = 0
    while first < len(words):
        start, end = words[first].start(), first
        while end < min(len(words), first + max_words):
            if len(text[start:words[end].end()].encode("utf-8")) > max_bytes:
                break
            end += 1
        if end == first:
            raise ValueError("A source token exceeds the configured passage byte budget")
        yield start, words[end - 1].end()
        if end == len(words):
            break
        first = max(first + 1, end - overlap_words)


# Keep raw speaker tokens; these descriptions are provenance, never person resolution.
ROLE_DESCRIPTIONS = {
    "CDR": "Commander (James A. Lovell)", "CMP": "Command module pilot (John L. Swigert)",
    "LMP": "Lunar module pilot (Fred W. Haise)", "CC": "CapCom role; not a fixed person",
    "SC": "Unidentified spacecraft speaker", "MS": "Multiple speakers",
}
SPEAKER_TOKEN = r"(?P<speaker>[A-Za-z][A-Za-z0-9*/-]*|\[[^\]\r\n]+\])(?=[ \t\r\n.,:;!?]|$)"
PRIMARY_MARKER = re.compile(r"(?m)^[ \t]*(?P<met>\d{3}:[0-5]\d:[0-5]\d)[ \t]+" + SPEAKER_TOKEN)
INLINE_MARKER = re.compile(r"(?<!\d)(?P<days>\d{2})[ \t]+(?P<hours>[0-2]\d)[ \t]+(?P<minutes>[0-5]\d)[ \t]+(?P<seconds>[0-5]\d)[ \t]+" + SPEAKER_TOKEN)


def transcript_turns(text):
    markers = [(m.start(), m.end(), m.group("met"), m.group("speaker"), "primary", m.start("speaker")) for m in PRIMARY_MARKER.finditer(text)]
    for m in INLINE_MARKER.finditer(text):
        if int(m.group("hours")) > 23:
            continue
        met = f"{int(m.group('days')) * 24 + int(m.group('hours')):03d}:{m.group('minutes')}:{m.group('seconds')}"
        markers.append((m.start(), m.end(), met, m.group("speaker"), "inline", m.start("speaker")))
    markers.sort()
    turns, previous = [], -1
    if markers and text[:markers[0][0]].strip():
        turns.append({"start_char": 0, "end_char": markers[0][0], "text": text[:markers[0][0]], "parse_warnings": ["Text before first recognized timestamp"], "speaker_role": "unknown"})
    for index, (start, body, met, speaker, marker_kind, speaker_start) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        seconds = sum(int(v) * scale for v, scale in zip(met.split(":"), (3600, 60, 1)))
        warnings = []
        if seconds < previous:
            warnings.append("Nonmonotone source timestamp retained without correction")
        previous = seconds
        role = re.split(r"[-/]", speaker.upper())[0]
        if role not in ROLE_DESCRIPTIONS:
            warnings.append(f"Unknown raw speaker token: {speaker}")
            # Unknown marker tokens may be utterance words: retain them as source text.
            body = speaker_start
        turn = {"start_char": body, "end_char": end, "text": text[body:end], "met": met, "speaker": speaker,
                "speaker_role": role if role in ROLE_DESCRIPTIONS else "unknown", "marker_kind": marker_kind}
        if role in ROLE_DESCRIPTIONS:
            turn["role_description"] = ROLE_DESCRIPTIONS[role]
        if warnings:
            turn["parse_warnings"] = warnings
        turns.append(turn)
    if not markers:
        return [{"start_char": 0, "end_char": len(text), "text": text, "speaker_role": "unknown", "parse_warnings": ["No recognized transcript markers"]}]
    return turns


def source_slices(text, max_chars, turns):
    """Bound slices at supplied turns; retained for exact-offset source tooling."""
    for turn in turns or [{"start_char": 0, "end_char": len(text)}]:
        start, end = turn["start_char"], turn["end_char"]
        while start < end:
            stop = min(start + max_chars, end)
            if stop < end:
                boundary = text.rfind(" ", start, stop)
                if boundary > start:
                    stop = boundary
            yield start, stop
            start = stop


def _segments(row, turns=None):
    text, kind = row["text"], row["source_kind"]
    base = {"kind": kind, "title": row["title"]}
    if kind == "mission_transcript":
        for turn in turns if turns is not None else transcript_turns(text):
            context = {**base, **{k: v for k, v in turn.items() if k not in {"start_char", "end_char", "text", "marker_kind"}}}
            yield turn["start_char"], turn["end_char"], context
    elif kind == "review_board":
        # OCR contains contents-page references and damaged/missing footers.
        # Cite a printed label only between consecutive uncorrected labels in
        # the report body. Ambiguous gaps retain offsets/source URL alone.
        body = re.search(r"(?m)^CHAPTER[ \t]*\n[ \t\n]*[iI1][ \t]*\n[ \t\n]*AUTHORITIES", text)
        body_start = body.start() if body else len(text)
        labels = list(re.finditer(r"(?m)^[ \t]*([1-5])[ \t]*[-–][ \t]*(\d{1,3})[ \t]*$", text))
        start, previous = 0, None
        for match in labels:
            end = match.end()
            label = (int(match.group(1)), int(match.group(2)))
            context = base
            if start >= body_start and previous is not None and label == (previous[0], previous[1] + 1):
                context = {**base, "printed_label": f"{label[0]}-{label[1]}"}
            yield start, end, context
            start, previous = end, label
        if start < len(text):
            yield start, len(text), base
    elif row.get("sections"):
        for section in row["sections"]:
            start, end = section["start_char"], section["end_char"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid section offsets: {row['id']}")
            yield start, end, base
    else:
        yield 0, len(text), base


def prepare_inputs(cfg, max_passages=None):
    cfg = _data(cfg)
    if max_passages is not None and (type(max_passages) is not int or max_passages < 1):
        raise ValueError("max_passages must be a positive integer")
    rows = read_manifest(cfg)
    sources = {r["id"]: {k: v for k, v in r.items() if k != "text"} for r in rows}
    passages = []
    for row in rows:
        turns = transcript_turns(row["text"]) if row["source_kind"] == "mission_transcript" else None
        if turns is not None:
            sources[row["id"]]["transcript_parse"] = {
                "primary_markers": sum(t.get("marker_kind") == "primary" for t in turns),
                "inline_markers": sum(t.get("marker_kind") == "inline" for t in turns),
                "warnings": [{"start_char": t["start_char"], "warnings": t["parse_warnings"]}
                             for t in turns if t.get("parse_warnings")],
                "empty_turns": [{k: v for k, v in t.items() if k != "text"}
                                for t in turns if not t["text"].strip()],
            }
        for begin, end, context in _segments(row, turns):
            for local_start, local_end in windows(row["text"][begin:end], **cfg["passages"]):
                if max_passages is not None and len(passages) >= max_passages:
                    break
                start, stop = begin + local_start, begin + local_end
                text = row["text"][start:stop]
                passages.append({"id": "passage:" + digest([row["id"], row["sha256"], start, stop, text])[:24],
                                 "source_id": row["id"], "source_start": start, "source_end": stop,
                                 "text": text, "context": context})
            if max_passages is not None and len(passages) >= max_passages:
                break
        if max_passages is not None and len(passages) >= max_passages:
            break
    if not passages:
        raise ValueError("No nonempty passages selected")
    return {"sources": sources, "passages": passages,
            "corpus_sha256": digest([{k: v for k, v in r.items() if k != "text"} for r in rows]),
            "selection": {"max_passages": max_passages, "partial": max_passages is not None,
                          "sources": len(sources), "passages": len(passages)}}
