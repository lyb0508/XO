"""Uvicorn entrypoint for container platforms that ignore --factory."""

from app.api import create_app

app = create_app()
