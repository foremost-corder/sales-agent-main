import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import sys

from sqlalchemy import select

from sales_agent.domain.models import Call
from sales_agent.core.database import SessionLocal
from sales_agent.features.calls.import_workflow import CallImportParseError, parse_call_file


def current_week_dates() -> list[date]:
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return [monday + timedelta(days=offset) for offset in range(7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or import a folder tree of sales call TXT files."
    )
    parser.add_argument("source", type=Path, help="Root folder containing TXT files")
    parser.add_argument("--commit", action="store_true", help="Write valid calls to PostgreSQL")
    parser.add_argument("--user-id", default="local-demo-user")
    parser.add_argument("--sales-id", default="001")
    parser.add_argument("--sales-stage", default="销售线索")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    root = args.source.resolve()
    if not root.is_dir():
        print(f"ERROR source folder does not exist: {root}")
        return 2

    files = sorted(root.rglob("*.txt"), key=lambda path: path.as_posix().casefold())
    fallback_dates = current_week_dates()
    parsed_files = []
    failures: list[tuple[str, str]] = []
    seen_external_ids: set[str] = set()

    for index, path in enumerate(files):
        try:
            parsed = parse_call_file(
                path,
                root=root,
                fallback_date=fallback_dates[index % len(fallback_dates)],
            )
        except (CallImportParseError, OSError) as exc:
            failures.append((path.relative_to(root).as_posix(), str(exc)))
            continue
        if parsed.external_call_id in seen_external_ids:
            failures.append((parsed.source_filename, "duplicate filename stem in batch"))
            continue
        seen_external_ids.add(parsed.external_call_id)
        parsed_files.append(parsed)

    with SessionLocal() as session:
        existing_ids = set(
            session.scalars(
                select(Call.external_call_id).where(Call.user_id == args.user_id)
            ).all()
        )

    new_files = [item for item in parsed_files if item.external_call_id not in existing_ids]
    duplicate_count = len(parsed_files) - len(new_files)

    print(f"Mode: {'COMMIT' if args.commit else 'DRY RUN'}")
    print(f"Source: {root}")
    print(f"TXT files found: {len(files)}")
    print(f"Valid call files: {len(parsed_files)}")
    print(f"New calls: {len(new_files)}")
    print(f"Existing calls skipped: {duplicate_count}")
    print(f"Invalid/non-call files skipped: {len(failures)}")
    print(f"Sales ID: {args.sales_id}")
    print("Call dates:")
    for call_date, count in sorted(Counter(item.call_date for item in new_files).items()):
        print(f"  {call_date.isoformat()}: {count}")

    if failures:
        print("Skipped files:")
        for filename, reason in failures:
            print(f"  {filename}: {reason}")

    if not args.commit:
        print("No database rows were written. Re-run with --commit to import.")
        return 0

    if not new_files:
        print("Nothing new to import.")
        return 0

    with SessionLocal.begin() as session:
        session.add_all(
            [
                Call(
                    user_id=args.user_id,
                    external_call_id=item.external_call_id,
                    sales_id=args.sales_id,
                    call_date=item.call_date,
                    sales_stage=args.sales_stage,
                    source_filename=item.source_filename,
                    source_encoding=item.source_encoding,
                    raw_source_text=item.raw_source_text,
                    transcript_text=item.transcript_text,
                    source_hash=item.source_hash,
                    metadata_is_synthetic=True,
                    analysis_status="pending",
                )
                for item in new_files
            ]
        )

    print(f"Imported {len(new_files)} calls successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
