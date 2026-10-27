import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.io import load_config, to_display
from meghmukt.masking import CLOUD, SHADOW, MaskParams, detect, detect_deep, scl_to_mask


def main():
    cfg = load_config()
    P = cfg["paths"]
    params = MaskParams(**json.loads((P["processed"] / "mask_params.json").read_text())["params"])
    methods = {"rules": lambda x: detect(x, params)[0], "deep": lambda x: detect_deep(x)[0]}
    held = cfg["split"]["test"] + cfg["split"]["val"]
    stats = {m: {c: [0, 0, 0] for c in ("cloud", "shadow", "both")} | {"false_alarm": []} for m in methods}
    examples = []
    for aoi in held:
        d = np.load(P["processed"] / f"s2_{aoi}.npz")
        for x, s in zip(d["cloudy"].astype(np.float32), d["scl"]):
            ref = scl_to_mask(s)
            preds = {m: f(x) for m, f in methods.items()}
            examples.append((aoi, x, ref, preds))
            for m, det in preds.items():
                for c, classes in (("cloud", (CLOUD,)), ("shadow", (SHADOW,)), ("both", (CLOUD, SHADOW))):
                    r, p = np.isin(ref, classes), np.isin(det, classes)
                    st = stats[m][c]
                    st[0] += int((r & p).sum()); st[1] += int((~r & p).sum()); st[2] += int((r & ~p).sum())
        for x in d["clear"].astype(np.float32):
            for m, f in methods.items():
                stats[m]["false_alarm"].append(float((f(x) != 0).mean()))
    report = {}
    for m in methods:
        report[m] = {}
        for c in ("cloud", "shadow", "both"):
            tp, fp, fn = stats[m][c]
            report[m][c] = {"f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
                            "recall": round(tp / max(tp + fn, 1), 3)}
        report[m]["false_alarm_clear_dates"] = round(float(np.mean(stats[m]["false_alarm"])), 4)
        print(f"{m:6s} cloud F1 {report[m]['cloud']['f1']:.3f}  shadow F1 {report[m]['shadow']['f1']:.3f}  "
              f"cloud+shadow F1 {report[m]['both']['f1']:.3f}  false alarms on clear dates {report[m]['false_alarm_clear_dates']:.2%}")
    (P["results"] / "mask_benchmark.json").write_text(json.dumps(
        {"areas": held, "reference": "ESA SCL (8/9/10 cloud, 3 shadow) — imperfect for shadow/haze", "results": report}, indent=2))

    liss = np.load(P["processed"] / "liss4.npz")["cloudy_patch"]
    rows = [(f"{a} (Sentinel-2)", x, ref, p) for a, x, ref, p in examples[:2]]
    rows.append(("real LISS-IV patch", liss, None, {"rules": detect(liss, params)[0], "deep": detect_deep(liss, 5.0)[0]}))
    fig, ax = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)))
    for r, (t, x, ref, p) in enumerate(rows):
        ax[r, 0].imshow(to_display(x)), ax[r, 0].set_ylabel(t)
        for c, (title, m) in enumerate((("ESA SCL label", ref), ("MeghMukt rules", p["rules"]), ("OmniCloudMask (deep)", p["deep"])), 1):
            if m is None:
                ax[r, c].text(0.5, 0.5, "no label for LISS-IV", ha="center", va="center", transform=ax[r, c].transAxes)
            else:
                ax[r, c].imshow(m, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
            ax[r, c].set_title(title if r == 0 else "")
        ax[r, 0].set_title("input (false colour)" if r == 0 else "")
        for a in ax[r]:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Cloud (teal) and shadow (yellow) masks on unseen scenes", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(P["results"] / "figures" / "mask_benchmark.png", dpi=100)


if __name__ == "__main__":
    main()
