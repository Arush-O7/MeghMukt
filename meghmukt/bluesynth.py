from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ratio_features(X: np.ndarray) -> np.ndarray:
    g, r, n = X
    eps = 1e-6
    ndvi = (n - r) / (n + r + eps)
    ndwi = (g - n) / (g + n + eps)
    rg = np.clip(r / (g + eps), 0, 3)
    ng = np.clip(n / (g + eps), 0, 10)
    base = [ndvi, ndwi, rg, ng]
    feats = [np.ones_like(g)] + base + [a * b for i, a in enumerate(base) for b in base[i:]]
    return np.stack(feats)


class BlueSynth:
    def __init__(self, kind: str, params: dict):
        self.kind, self.p = kind, params

    @classmethod
    def fit(cls, X: np.ndarray, blue: np.ndarray, kind: str = "ratio") -> "BlueSynth":
        if kind == "ratio":
            F = ratio_features(X)
            y = blue / (X[0] + 1e-6)
            w, *_ = np.linalg.lstsq(F.T, y, rcond=None)
            return cls("ratio", {"w": w.tolist()})
        from sklearn.neural_network import MLPRegressor
        F = cls._mlp_feats(X)
        mu, sd = F.mean(1), F.std(1) + 1e-6
        m = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=300, early_stopping=True, random_state=0)
        m.fit(((F.T - mu) / sd), blue)
        return cls("mlp", {"mu": mu.tolist(), "sd": sd.tolist(),
                           "W": [c.tolist() for c in m.coefs_], "b": [c.tolist() for c in m.intercepts_]})

    @staticmethod
    def _mlp_feats(X):
        g, r, n = X
        return np.stack([g, r, n, (n - r) / (n + r + 1e-6), (g - n) / (g + n + 1e-6)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        shape = X.shape[1:]
        X = X.reshape(3, -1)
        if self.kind == "ratio":
            out = X[0] * np.clip(np.asarray(self.p["w"]) @ ratio_features(X), 0.2, 2.0)
        else:
            h = (self._mlp_feats(X).T - np.asarray(self.p["mu"])) / np.asarray(self.p["sd"])
            for i, (W, b) in enumerate(zip(self.p["W"], self.p["b"])):
                h = h @ np.asarray(W) + np.asarray(b)
                if i < len(self.p["W"]) - 1:
                    h = np.maximum(h, 0)
            out = h.ravel()
        return np.clip(out, 0, None).reshape(shape).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({"kind": self.kind, "params": self.p}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["kind"], d["params"])
