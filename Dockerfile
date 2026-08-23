# Runtime image for the industrial diagnostic agent HTTP API.
# The graph talks to a local Ollama endpoint; inside Docker point
# INDUSTRIAL_AGENT_OLLAMA_BASE_URL at host.docker.internal:11434.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml pylock.toml README.md ./
COPY app ./app
RUN python -m pip install --no-cache-dir .

# Non-root runtime user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# create_app is an application factory; --factory makes uvicorn call it.
CMD ["uvicorn", "app.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
