from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
import re

SPEAKER_LINE_PATTERN = re.compile(r"(?m)^\s*用户\d+\s*[：:]")
SEPARATOR_PATTERN = re.compile(r"^\s*-{3,}\s*$")


class CallImportParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedCallFile:
    external_call_id: str
    call_date: date
    source_filename: str
    source_encoding: str
    raw_source_text: str
    transcript_text: str
    source_hash: str


def decode_call_file(raw_bytes: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise CallImportParseError("unsupported text encoding")


def extract_transcript(raw_text: str) -> str:
    match = SPEAKER_LINE_PATTERN.search(raw_text)
    if match is None:
        raise CallImportParseError("no speaker line found")

    lines = raw_text[match.start() :].splitlines()
    while lines and (not lines[-1].strip() or SEPARATOR_PATTERN.fullmatch(lines[-1])):
        lines.pop()
    transcript = "\n".join(lines).strip()
    if not transcript:
        raise CallImportParseError("empty transcript")
    return transcript


def date_from_parent(path: Path, fallback: date) -> date:
    try:
        return date.fromisoformat(path.parent.name)
    except ValueError:
        return fallback


def parse_call_file(path: Path, *, root: Path, fallback_date: date) -> ParsedCallFile:
    raw_bytes = path.read_bytes()
    raw_text, encoding = decode_call_file(raw_bytes)
    transcript = extract_transcript(raw_text)
    return ParsedCallFile(
        external_call_id=path.stem,
        call_date=date_from_parent(path, fallback_date),
        source_filename=path.relative_to(root).as_posix(),
        source_encoding=encoding,
        raw_source_text=raw_text,
        transcript_text=transcript,
        source_hash=sha256(raw_bytes).hexdigest(),
    )

