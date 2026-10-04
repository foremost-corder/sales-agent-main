"""Deterministically parse transcript text already stored on a Call row."""

import re

from sales_agent.features.knowledge.contracts import SpeakerRole, TranscriptTurn


PARSER_VERSION = "db-transcript-turns-v1"
TURN_PATTERN = re.compile(
    r"^\s*(?P<label>用户(?P<number>\d+))\s*[：:]\s*"
    r"(?P<body>.*?)"
    r"(?:[，,]\s*(?P<time>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?))?\s*$"
)


class TranscriptParseError(ValueError):
    pass


def parse_transcript_turns(transcript_text: str) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    for raw_line in transcript_text.splitlines():
        if not raw_line.strip():
            continue
        match = TURN_PATTERN.fullmatch(raw_line)
        if match is None:
            raise TranscriptParseError(f"unrecognized transcript line: {raw_line[:120]}")
        body = match.group("body").strip().rstrip("，,").strip()
        if not body:
            continue
        turns.append(
            TranscriptTurn(
                turn_no=len(turns) + 1,
                source_speaker_label=match.group("label"),
                # Diarization labels are not business roles. A separate model
                # resolves every observed label once for the whole call.
                speaker=SpeakerRole.UNKNOWN,
                timestamp_ms=_timestamp_ms(match.group("time")),
                text=body,
                raw_line=raw_line,
            )
        )
    if not turns:
        raise TranscriptParseError("transcript contains no valid turns")
    return turns


def _timestamp_ms(value: str | None) -> int | None:
    if value is None:
        return None
    hour, minute, second = value.split(":")
    return int((int(hour) * 3600 + int(minute) * 60 + float(second)) * 1000)
