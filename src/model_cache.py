from policy_value_net_numpy import PolicyValueNetNumpy
import pickle
import threading


class _ModelCache(object):
    """
    A cache for policy value networks.
    """

    _lock: threading.Lock
    _models: dict[tuple[str, int, int], PolicyValueNetNumpy]

    def __init__(self):
        self._lock = threading.Lock()
        self._models = {}

    def get(self, model_file, width, height):
        key = (model_file, width, height)
        with self._lock:
            if key in self._models:
                return self._models[key]
            policy_param = pickle.load(open(model_file, "rb"), encoding="bytes")
            model = PolicyValueNetNumpy(width, height, policy_param)
            self._models[key] = model
            return model


MODEL_CACHE = _ModelCache()
