import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sales_agent.api.conversations import router as conversations_router
from sales_agent.core.config import get_settings
from sales_agent.core.database import database_is_ready

settings = get_settings()
logging.getLogger("sales_agent").setLevel(logging.INFO)
web_directory = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(conversations_router)
app.mount("/static", StaticFiles(directory=web_directory), name="static")


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(web_directory / "index.html")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/database", tags=["health"])
def database_health() -> dict[str, str]:
    if not database_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database unavailable",
        )
    return {"status": "ok", "database": "reachable"}


@app.get("/health/model", tags=["health"])
def model_health() -> dict[str, str | bool]:
    return {
        "status": "configured" if settings.has_openai_api_key else "configuration_required",
        "configured": settings.has_openai_api_key,
        "model": settings.chat_model,
    }
