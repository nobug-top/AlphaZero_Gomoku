install:
    uv sync
train:
    just install
    uv run src/train.py