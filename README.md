## Gomoku MCTS (AlphaZero style)

This is a minimal Gomoku (Gobang) project that keeps an AlphaZero-style MCTS
implementation for human vs AI play.

### Requirements

- Python >= 3.8
- Numpy >= 1.11

### Getting Started

Run the following script from the directory:

```
python human_play.py
```

### HTTP Service

Start a simple HTTP service for inference:

```
python http_server.py
```

Endpoints:

- `GET /health`
- `GET /schema`
- `POST /infer`

Example request:

```
curl -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "current_player": 1,
    "last_move": -1,
    "points": [
      {"x": 0, "y": 0, "player": 1},
      {"x": 0, "y": 1, "player": 2}
    ]
  }'
```

### Parameters

The game and MCTS parameters are fixed in `human_play.py` via
`get_game_params()` and are used by `http_server.py` for inference.

### Notes

- `mcts_pure.py` is kept for reference but is not wired into `human_play.py`.
