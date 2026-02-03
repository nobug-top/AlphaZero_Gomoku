#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple HTTP service for MCTS inference.
"""

from __future__ import print_function
from model_cache import MODEL_CACHE

import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from game import Board
from mcts_alphaZero import MCTSPlayer


def _build_board(params, state):
    board = Board(
        width=params["width"], height=params["height"], n_in_row=params["n_in_row"]
    )
    board.init_board(start_player=0)

    points = state.get("points")
    if points is None:
        raise ValueError("missing points in request")
    states = {}
    for item in points:
        x = int(item["x"])
        y = int(item["y"])
        player = int(item["player"])
        move = board.location_to_move([x, y])
        if move == -1:
            raise ValueError("invalid point ({}, {})".format(x, y))
        states[move] = player

    board.states = states
    board.availables = [i for i in range(board.width * board.height) if i not in states]
    board.last_move = int(state.get("last_move", -1))
    current_player = state.get("current_player")
    if current_player in board.players:
        board.current_player = current_player
    return board


def _infer(payload):
    params = {
        "model_file": "best_policy_8_8_5.model",
        "width": 8,
        "height": 8,
        "n_in_row": 5,
        "c_puct": 5,
        "n_playout": 400,
        "temp": 1e-3,
    }
    state = payload
    board = _build_board(params, state)

    model = MODEL_CACHE.get(
        params["model_file"],
        params["width"],
        params["height"],
    )
    mcts_player = MCTSPlayer(
        model.policy_value_fn,
        c_puct=params["c_puct"],
        n_playout=params["n_playout"],
    )

    if len(board.availables) == 0:
        return {"error": "board is full"}

    move, move_probs = mcts_player.get_action(
        board,
        temp=params["temp"],
        return_prob=1,
    )
    return {
        "move": int(move),
        "location": board.move_to_location(int(move)),
        "move_probs": [float(x) for x in move_probs],
    }


class RequestHandler(BaseHTTPRequestHandler):
    def _check_auth(self):
        token = os.environ.get("API_TOKEN")
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == "Bearer {}".format(token)

    def _require_auth(self):
        body = json.dumps({"error": "unauthorized"}).encode("utf-8")
        self._last_response_size = len(body)
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self._last_response_size = len(body)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._last_request_size = 0
        self._request_start = time.time()
        if not self._check_auth():
            return self._require_auth()
        path = urlparse(self.path).path
        if path == "/health":
            return self._send_json(200, {"status": "ok"})
        if path == "/schema":
            return self._send_json(
                200,
                {
                    "state": {
                        "current_player": 1,
                        "last_move": -1,
                        "points": [
                            {"x": 0, "y": 0, "player": 1},
                            {"x": 0, "y": 1, "player": 2},
                        ],
                    }
                },
            )
        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        self._last_request_size = int(self.headers.get("Content-Length", "0"))
        self._request_start = time.time()
        if not self._check_auth():
            return self._require_auth()
        path = urlparse(self.path).path
        if path != "/infer":
            return self._send_json(404, {"error": "not found"})

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8")
        self._last_request_size = len(raw.encode("utf-8"))
        try:
            payload = json.loads(raw) if raw else {}
        except Exception as exc:
            return self._send_json(400, {"error": "invalid json", "detail": str(exc)})

        try:
            result = _infer(payload)
        except Exception as exc:
            print("ERROR: infer failed:", exc, flush=True)
            traceback.print_exc()
            return self._send_json(500, {"error": "infer failed", "detail": str(exc)})
        return self._send_json(200, result)

    def log_request(self, code="-", size="-"):
        req_size = getattr(self, "_last_request_size", 0)
        resp_size = getattr(self, "_last_response_size", 0)
        duration_ms = 0
        start = getattr(self, "_request_start", None)
        if start is not None:
            duration_ms = int((time.time() - start) * 1000)
        self.log_message(
            '"%s" %s req_bytes=%s resp_bytes=%s duration_ms=%s',
            self.requestline,
            str(code),
            req_size,
            resp_size,
            duration_ms,
        )

    def log_message(self, format, *args):
        message = "%s - - [%s] %s\n" % (
            self.client_address[0],
            self.log_date_time_string(),
            format % args,
        )
        print(message, end="", flush=True)


def run_server(host="0.0.0.0", port=8000):
    server = HTTPServer((host, port), RequestHandler)
    print("MCTS service listening on {}:{}".format(host, port), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    run_server(host=host, port=port)
