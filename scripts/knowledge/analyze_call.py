"""Run the call knowledge workflow directly, without the chat Agent."""

import argparse
import json
import logging
from uuid import UUID

from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.standard_tool import build_call_knowledge_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse one existing calls row and publish its vector knowledge."
    )
    parser.add_argument("call_id", type=UUID, help="Internal UUID from the calls table")
    parser.add_argument("--user-id", required=True, help="Owning application user ID")
    parser.add_argument("--force", action="store_true", help="Create a new analysis version even when the fingerprint matches")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    with SessionLocal() as session:
        result = build_call_knowledge_workflow(session).execute(
            call_id=args.call_id,
            user_id=args.user_id,
            force=args.force,
        )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
