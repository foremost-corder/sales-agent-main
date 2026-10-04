"""Analyze a filtered batch of calls concurrently, outside the chat Agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from threading import Lock
from time import perf_counter
from uuid import UUID

from sqlalchemy import select

from sales_agent.domain.models import Call
from sales_agent.integrations.embeddings.factory import build_embedding_provider
from sales_agent.core.config import get_settings
from sales_agent.core.database import SessionLocal
from sales_agent.features.knowledge.contracts import EmbeddingProvider
from sales_agent.features.knowledge.fact_extractor import OpenAIFactExtractor
from sales_agent.features.knowledge.lean_fact_extractor import OpenAILeanFactExtractor
from sales_agent.features.knowledge.namespaces import (
    BASELINE_KNOWLEDGE_NAMESPACE,
    LEAN_KNOWLEDGE_NAMESPACE,
    validate_knowledge_namespace,
)
from sales_agent.features.knowledge.speaker_resolver import OpenAISpeakerRoleResolver
from sales_agent.features.knowledge.standard_workflow import AnalyzeCallKnowledgeWorkflow
from sales_agent.features.knowledge.lean_workflow import AnalyzeCallKnowledgeLeanWorkflow


logger = logging.getLogger(__name__)


class _LockedEmbeddingProvider:
    """Reuse one local model safely while model-backed stages run concurrently."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._lock = Lock()
        self.provider_name = provider.provider_name
        self.model = provider.model
        self.dimensions = provider.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._provider.embed(texts)


@dataclass(frozen=True)
class CallTarget:
    call_id: UUID
    user_id: str
    external_call_id: str
    analysis_status: str


@dataclass(frozen=True)
class CallOutcome:
    call_id: str
    external_call_id: str
    ok: bool
    elapsed_seconds: float
    status: str
    reused: bool = False
    degraded: bool = False
    enrichment_status: str | None = None
    analysis_run_id: str | None = None
    fact_count: int | None = None
    low_confidence_fact_count: int | None = None
    document_count: int | None = None
    error_type: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze calls concurrently and publish their knowledge documents."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze every call in the database (cannot be combined with call selectors)",
    )
    parser.add_argument("--sales-id", help="Exact calls.sales_id value")
    parser.add_argument(
        "--call-date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Exact calls.call_date value",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        choices=range(1, 17),
        metavar="1-16",
        help="Number of concurrent call workers (default: 5)",
    )
    parser.add_argument(
        "--workflow",
        choices=("standard", "lean"),
        default="standard",
        help="Knowledge workflow to run (default: standard)",
    )
    parser.add_argument(
        "--namespace",
        dest="knowledge_namespace",
        help=(
            "Isolated knowledge namespace; defaults to baseline-react-v4 for standard "
            "and experiment-lean-v1 for lean"
        ),
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=("pending", "running", "completed", "failed"),
        dest="statuses",
        help="Only select this analysis status; repeat to include multiple statuses",
    )
    parser.add_argument("--limit", type=int, help="Analyze at most this many calls")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new analysis version even when the fingerprint matches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching calls without running the workflow",
    )
    return parser.parse_args()


def load_targets(
    *,
    sales_id: str | None,
    call_date: date | None,
    statuses: list[str] | None,
    limit: int | None,
) -> list[CallTarget]:
    statement = (
        select(
            Call.id,
            Call.user_id,
            Call.external_call_id,
            Call.analysis_status,
        )
        .order_by(Call.external_call_id, Call.id)
    )
    if sales_id is not None:
        statement = statement.where(Call.sales_id == sales_id)
    if call_date is not None:
        statement = statement.where(Call.call_date == call_date)
    if statuses:
        statement = statement.where(Call.analysis_status.in_(statuses))
    if limit is not None:
        statement = statement.limit(limit)
    with SessionLocal() as session:
        return [CallTarget(*row) for row in session.execute(statement).all()]


def analyze_target(
    target: CallTarget,
    *,
    force: bool,
    embedder: EmbeddingProvider,
    workflow_name: str,
    knowledge_namespace: str,
) -> CallOutcome:
    started = perf_counter()
    try:
        settings = get_settings()
        with SessionLocal() as session:
            if workflow_name == "lean":
                workflow = AnalyzeCallKnowledgeLeanWorkflow(
                    session,
                    extractor=OpenAILeanFactExtractor(settings),
                    embedder=embedder,
                )
            else:
                workflow = AnalyzeCallKnowledgeWorkflow(
                    session,
                    speaker_resolver=OpenAISpeakerRoleResolver(settings),
                    extractor=OpenAIFactExtractor(settings),
                    embedder=embedder,
                )
            result = workflow.execute(
                call_id=target.call_id,
                user_id=target.user_id,
                force=force,
                knowledge_namespace=knowledge_namespace,
            )
        return CallOutcome(
            call_id=str(target.call_id),
            external_call_id=target.external_call_id,
            ok=True,
            elapsed_seconds=round(perf_counter() - started, 3),
            status=result.status,
            reused=result.reused,
            degraded=result.degraded,
            enrichment_status=result.enrichment_status,
            analysis_run_id=result.analysis_run_id,
            fact_count=result.fact_count,
            low_confidence_fact_count=result.low_confidence_fact_count,
            document_count=result.document_count,
        )
    except Exception as exc:
        logger.exception(
            "batch_call_failed call_id=%s external_call_id=%s",
            target.call_id,
            target.external_call_id,
        )
        return CallOutcome(
            call_id=str(target.call_id),
            external_call_id=target.external_call_id,
            ok=False,
            elapsed_seconds=round(perf_counter() - started, 3),
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.all and (args.sales_id is not None or args.call_date is not None):
        raise SystemExit("--all cannot be combined with --sales-id or --call-date")
    if not args.all and args.sales_id is None and args.call_date is None:
        raise SystemExit(
            "select calls with --all, --sales-id, --call-date, or both selectors"
        )
    try:
        knowledge_namespace = validate_knowledge_namespace(
            args.knowledge_namespace
            or (
                LEAN_KNOWLEDGE_NAMESPACE
                if args.workflow == "lean"
                else BASELINE_KNOWLEDGE_NAMESPACE
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    targets = load_targets(
        sales_id=args.sales_id,
        call_date=args.call_date,
        statuses=args.statuses,
        limit=args.limit,
    )
    status_counts: dict[str, int] = {}
    for target in targets:
        status_counts[target.analysis_status] = status_counts.get(target.analysis_status, 0) + 1
    print(
        json.dumps(
            {
                "event": "batch_selected",
                "scope": "all" if args.all else "filtered",
                "sales_id": args.sales_id,
                "call_date": args.call_date.isoformat() if args.call_date else None,
                "call_count": len(targets),
                "status_counts": status_counts,
                "status_filter": args.statuses,
                "workers": args.workers,
                "workflow": args.workflow,
                "knowledge_namespace": knowledge_namespace,
                "force": args.force,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not targets or args.dry_run:
        return 0

    settings = get_settings()
    base_embedder = build_embedding_provider(settings)
    embedder: EmbeddingProvider = (
        _LockedEmbeddingProvider(base_embedder)
        if base_embedder.provider_name == "fastembed"
        else base_embedder
    )
    started = perf_counter()
    outcomes: list[CallOutcome] = []
    with ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="call-knowledge",
    ) as executor:
        futures: dict[Future[CallOutcome], CallTarget] = {
            executor.submit(
                analyze_target,
                target,
                force=args.force,
                embedder=embedder,
                workflow_name=args.workflow,
                knowledge_namespace=knowledge_namespace,
            ): target
            for target in targets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outcome = future.result()
            outcomes.append(outcome)
            print(
                json.dumps(
                    {
                        "event": "call_completed",
                        "progress": f"{completed}/{len(targets)}",
                        **asdict(outcome),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    outcomes.sort(key=lambda item: item.external_call_id)
    successful = [item for item in outcomes if item.ok]
    failed = [item for item in outcomes if not item.ok]
    print(
        json.dumps(
            {
                "event": "batch_completed",
                "elapsed_seconds": round(perf_counter() - started, 3),
                "success_count": len(successful),
                "failed_count": len(failed),
                "reused_count": sum(item.reused for item in successful),
                "created_count": sum(not item.reused for item in successful),
                "degraded_count": sum(item.degraded for item in successful),
                "fact_count": sum(item.fact_count or 0 for item in successful),
                "low_confidence_fact_count": sum(
                    item.low_confidence_fact_count or 0 for item in successful
                ),
                "document_count": sum(item.document_count or 0 for item in successful),
                "failures": [asdict(item) for item in failed],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
