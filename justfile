docker := if `command -v podman` != '1' { "podman" } else { "docker" }

install:
    uv sync

dev:
    just install
    uv run src/http_server.py

docker-build:
    {{ docker }} build -t alphazero-gomoku:latest .
