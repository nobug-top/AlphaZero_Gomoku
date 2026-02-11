from policy_value_net_pytorch import PolicyValueNet
import threading


class _ModelCache(object):
    """
    A cache for policy value networks.
    """

    _lock: threading.Lock
    _models: dict[str, PolicyValueNet]

    def __init__(self):
        self._lock = threading.Lock()
        self._models = {}

    def get(self, model_file, board_width, board_height) -> PolicyValueNet:
        with self._lock:
            if model_file in self._models:
                return self._models[model_file]
            model = PolicyValueNet(board_width, board_height, model_file)
            self._models[model_file] = model
            return model


MODEL_CACHE = _ModelCache()
