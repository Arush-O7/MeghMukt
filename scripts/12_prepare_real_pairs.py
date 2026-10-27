import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meghmukt.dataset import Normaliser, make_condition
from meghmukt.io import Harmoniser, load_config
from meghmukt.masking import detect_auto

spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "11_eval_real_pairs.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
VAL_AREAS = ["silchar"]


def build(meta_paths, harm):
    out = []
    for mp in meta_paths:
        meta = json.loads(mp.read_text())
        if not meta.get("pairs") or not meta.get("reference"):
            continue
        ref, _ = ev.load(mp.parent / meta["reference"], harm)
        for pr in meta["pairs"]:
            cloudy, _ = ev.load(mp.parent / pr["cloudy"], harm)
            truth, scl_t = ev.load(mp.parent / pr["truth"], harm)
            valid = ~np.isin(scl_t, (0, 1, 3, 8, 9, 10))
            mask = detect_auto(cloudy, "deep", 10.0)[0]
            both = (mask == 0) & valid
            truth_m = ev.match(truth, cloudy, both)
            corr = float(np.mean([np.corrcoef(truth_m[b][both], cloudy[b][both])[0, 1] for b in range(3)]))
            keep = corr >= ev.MIN_CORR and ((mask > 0) & valid).mean() > 0.05
            print(f"  {meta['aoi']:10s} {pr['cloudy'][:10]} corr {corr:.2f} cloud {(mask > 0).mean():.0%} {'kept' if keep else 'DROPPED'}")
            if keep:
                out.append(dict(cloudy=cloudy, mask=mask, truth=truth_m, valid=valid, ref=ref, name=f"{meta['aoi']} {pr['cloudy'][:10]}"))
    return out


def main():
    cfg = load_config()
    P = cfg["paths"]
    harm = Harmoniser.load(P["processed"] / "harmoniser.json")
    norm = Normaliser.load(P["processed"] / "norm.json")

    print("[real] training areas")
    tr = build(sorted((P["raw"] / "real_pairs_train").glob("*/meta.json")), harm)
    np.savez_compressed(P["processed"] / "realpairs_train.npz",
                        cloudy=np.stack([p["cloudy"] for p in tr]).astype(np.float16),
                        mask=np.stack([p["mask"] for p in tr]), truth=np.stack([p["truth"] for p in tr]).astype(np.float16),
                        valid=np.stack([p["valid"] for p in tr]), ref=np.stack([p["ref"] for p in tr]).astype(np.float16),
                        names=np.array([p["name"] for p in tr]))
    print(f"[real] {len(tr)} training pairs -> realpairs_train.npz")

    print("[real] validation area")
    va = build([P["raw"] / "real_pairs" / a / "meta.json" for a in VAL_AREAS], harm)
    conds, gts, vals = [], [], []
    rng = np.random.default_rng(5)
    for p in va:
        H = p["mask"].shape[0]
        n = 0
        while n < 24:
            y, x = rng.integers(0, H - 128, 2)
            m = p["mask"][y:y + 128, x:x + 128]
            v = p["valid"][y:y + 128, x:x + 128]
            if not 0.1 < (m > 0).mean() < 0.95 or v.mean() < 0.8:
                continue
            conds.append(make_condition(p["cloudy"][:, y:y + 128, x:x + 128], m, p["ref"][:, y:y + 128, x:x + 128], norm, True))
            gts.append(norm(p["truth"][:, y:y + 128, x:x + 128]))
            vals.append(v)
            n += 1
    np.savez_compressed(P["processed"] / "eval_real_val.npz", cond=np.stack(conds).astype(np.float16),
                        gt=np.stack(gts).astype(np.float16), valid=np.stack(vals),
                        meta=json.dumps({"areas": VAL_AREAS, "note": "real clouds, near-date truth"}))
    print(f"[real] {len(conds)} validation crops from {len(va)} real pairs -> eval_real_val.npz")


if __name__ == "__main__":
    main()
