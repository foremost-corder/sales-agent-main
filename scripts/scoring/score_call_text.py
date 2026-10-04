"""Score one transcript TXT with the independent whiteboard workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pydantic import ValidationError

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.core.config import get_settings
from sales_agent.features.knowledge.parser import TranscriptParseError
from sales_agent.features.single_call_scoring.model import (
    OpenAICallLevelClassifier,
    OpenAIEvidenceReviewer,
    OpenAIModuleJudge,
    WhiteboardModelOutputError,
)
from sales_agent.features.single_call_scoring.workflow import (
    SingleCallWhiteboardScoringWorkflow,
    WhiteboardScoringError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify one transcript TXT, score applicable modules in parallel, "
            "review evidence, and emit the 14-dimension whiteboard."
        )
    )
    parser.add_argument("transcript", type=Path, help="TXT transcript path")
    parser.add_argument("--call-id", help="Output call id; defaults to the file stem")
    parser.add_argument("--sales-stage", default="销售线索")
    parser.add_argument("--model", help="Override SCORING_MODEL for this run")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        transcript_text = _read_transcript(args.transcript)
        settings = get_settings()
        if args.model:
            settings = settings.model_copy(update={"scoring_model": args.model})
        result = SingleCallWhiteboardScoringWorkflow(
            OpenAICallLevelClassifier(settings),
            OpenAIModuleJudge(settings),
            OpenAIEvidenceReviewer(settings),
        ).score(
            call_id=args.call_id or args.transcript.stem,
            sales_stage=args.sales_stage,
            transcript_text=transcript_text,
        )
    except (
        OSError,
        ValidationError,
        ChatModelNotConfiguredError,
        ChatModelRequestError,
        TranscriptParseError,
        WhiteboardModelOutputError,
        WhiteboardScoringError,
    ) as exc:
        error = {
            "schema_version": "single-call-whiteboard-error-v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    rendered = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"whiteboard score written to {args.output.resolve()}")
    else:
        print(rendered)
    return 0


def _read_transcript(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode transcript as UTF-8 or GB18030: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
