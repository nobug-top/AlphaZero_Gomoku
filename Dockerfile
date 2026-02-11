FROM python:3.12-slim as builder
# install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --compile-bytecode --no-dev

FROM python:3.12-slim as runner

# Create non-root user for running the app
RUN addgroup --system appgroup && adduser --system appuser --ingroup appgroup
USER appuser

COPY --from=builder /app/.venv /app/.venv
WORKDIR /app

COPY best_policy_8_8_5.model .
COPY best_policy_10_10_5.model .
COPY src/ .

CMD ["/app/.venv/bin/python", "http_server.py"]