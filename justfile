docker := if `command -v podman` != '1' { "podman" } else { "docker" }

install:
    uv sync

dev:
    just install
    BOARD_SIZE=10 uv run src/http_server.py

docker-build:
    {{ docker }} build -t alphazero-gomoku:latest .

docker-compose-up:
    {{ docker }} compose up --build --force-recreate

docker-compose-down:
    {{ docker }} compose down -v