import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meghmukt.dataset import Normaliser, SyntheticCloudDataset, make_condition
from meghmukt.io import load_config
from meghmukt.synth import CloudSynthesiser

CLEAR_MAX, MIN_CORR = 0.03, 0.8
PAIRS = [("2024-05-11", "2024-05-23", "2023-11-13"), ("2023-10-20", "2023-11-01", "2024-11-19")]


def match(src, dst, ok):
    out = src.copy()
    for b in range(3):
        x, y = src[b][ok], dst[b][ok]
        a = np.cov(x, y, bias=True)[0, 1] / max(x.var(), 1e-9)
        out[b] = a * src[b] + (y.mean() - a * x.mean())
    return out


def real_pairs(d, idx):
    dates = list(d["dates"])
    X, M, F = d["X"], d["mask"], d["cloud_frac"]
    out = []
    for cloudy_d, truth_d, ref_d in PAIRS:
        ti, ri = dates.index(truth_d), dates.index(ref_d)
        if cloudy_d in dates:
            ci = dates.index(cloudy_d)
            src = {k: (X[k, ci], M[k, ci], F[k, ci]) for k in idx}
        else:
            ext = list(d[f"extraidx_{cloudy_d}"])
            src = {k: (d[f"extra_{cloudy_d}"][ext.index(k)], d[f"extramask_{cloudy_d}"][ext.index(k)],
                       float((d[f"extramask_{cloudy_d}"][ext.index(k)] > 0).mean())) for k in idx if k in ext}
        for k, (cx, cm, cf) in src.items():
            if not 0.05 < cf < 0.95 or F[k, ti] > CLEAR_MAX or F[k, ri] > CLEAR_MAX:
                continue
            cloudy, truth = cx.astype(np.float32), X[k, ti].astype(np.float32)
            valid = M[k, ti] == 0
            both = (cm == 0) & valid
            if both.mean() < 0.03:
                continue
            truth_m = match(truth, cloudy, both)
            corr = float(np.mean([np.corrcoef(truth_m[b][both], cloudy[b][both])[0, 1] for b in range(3)]))
            if corr < MIN_CORR:
                continue
            out.append(dict(cloudy=cloudy, mask=cm, truth=truth_m, valid=valid, ref=X[k, ri].astype(np.float32),
                            name=f"chip{k} {cloudy_d}->{truth_d}", corr=corr, cloud=cf))
    return out


def save_pairs(path, pairs):
    np.savez_compressed(path, cloudy=np.stack([p["cloudy"] for p in pairs]).astype(np.float16),
                        mask=np.stack([p["mask"] for p in pairs]), truth=np.stack([p["truth"] for p in pairs]).astype(np.float16),
                        valid=np.stack([p["valid"] for p in pairs]), ref=np.stack([p["ref"] for p in pairs]).astype(np.float16),
                        names=np.array([p["name"] for p in pairs]), corr=np.array([p["corr"] for p in pairs]),
                        cloud=np.array([p["cloud"] for p in pairs]))


def main():
    cfg = load_config()
    P = cfg["paths"]
    d = np.load(P["processed"] / "liss4_chips.npz")
    split, F, X = d["split"], d["cloud_frac"], d["X"]
    idx = {s: [int(k) for k in np.flatnonzero(split == s)] for s in ("train", "val", "test")}

    clear = [X[k, F[k] <= CLEAR_MAX] for k in idx["train"] if (F[k] <= CLEAR_MAX).sum() >= 2]
    np.savez_compressed(P["processed"] / "liss4_clear_train.npz", **{f"c{i}": c for i, c in enumerate(clear)})
    print(f"[liss4] {len(clear)} training chips with >= 2 clear dates "
          f"({sum(len(c) for c in clear)} clear chip-dates)")

    tr_pairs = real_pairs(d, idx["train"])
    save_pairs(P["processed"] / "realpairs_liss4_train.npz", tr_pairs)
    print(f"[liss4] {len(tr_pairs)} real LISS-IV cloud pairs in the training area:",
          ", ".join(f"{p['name'].split()[1]} ({p['cloud']:.0%})" for p in tr_pairs))

    norm = Normaliser.load(P["processed"] / "norm.json")
    synth = CloudSynthesiser(np.load(P["processed"] / "cloud_spectra.npy"), tuple(cfg["data"]["cloud_cover_range"]),
                             cfg["data"]["shadow_prob"])
    vs = [{"clear": X[k, F[k] <= CLEAR_MAX]} for k in idx["val"] if (F[k] <= CLEAR_MAX).sum() >= 2]
    ds = SyntheticCloudDataset(vs, norm, synth, patch=128, ref_dropout=0.0, seed=99, jitter=False)
    r = np.random.default_rng(3)
    items = [ds.sample(r) for _ in range(96)]
    np.savez_compressed(P["processed"] / "eval_liss4_val.npz", cond=np.stack([c for c, _ in items]).astype(np.float16),
                        gt=np.stack([g for _, g in items]).astype(np.float16), meta=json.dumps({"chips": idx["val"]}))
    print(f"[liss4] validation: {len(items)} synthetic-cloud crops from {len(vs)} validation chips")

    te_pairs = real_pairs(d, idx["test"])
    save_pairs(P["processed"] / "liss4_test_pairs.npz", te_pairs) if te_pairs else None
    np.savez_compressed(P["processed"] / "liss4_test_clear.npz",
                        **{f"c{i}": X[k, F[k] <= CLEAR_MAX] for i, k in enumerate(idx["test"]) if (F[k] <= CLEAR_MAX).sum() >= 2})
    print(f"[liss4] test: {len(idx['test'])} chips, {len(te_pairs)} real cloud pairs:",
          ", ".join(f"{p['name'].split()[1]} ({p['cloud']:.0%}, corr {p['corr']:.2f})" for p in te_pairs))


if __name__ == "__main__":
    main()
