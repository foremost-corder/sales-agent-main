import uvicorn

from sales_agent.api.app import app


def run() -> None:
    uvicorn.run("sales_agent.main:app", host="127.0.0.1", port=8000, reload=True)


__all__ = ["app", "run"]

