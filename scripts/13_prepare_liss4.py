import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meghmukt.io import load_config
from meghmukt.liss4 import BANDS, CrossCal, Scene
from meghmukt.masking import detect_auto

CHIP, MARGIN = 768, 24
REF_DATE = "2023-11-01"
CLEAR_MAX = 0.03
N_TRAIN, N_VAL, N_TEST = 48, 8, 200


def read_scene(sc: Scene, cal: CrossCal):
    bands, valid = [], None
    with rasterio.open(sc.band_path(3)) as s:
        transform = s.transform
    for b in BANDS:
        with rasterio.open(sc.band_path(b)) as s:
            dn = s.read(1)
        gain, off = sc.toa_factor(b)
        v = dn > 0
        valid = v if valid is None else valid & v
        i = BANDS.index(b)
        bands.append((np.clip(cal.a[i, 0, 0] * (dn.astype(np.float32) * gain + off) + cal.b[i, 0, 0], 0, 1)
                      ).astype(np.float16))
        del dn
    return np.stack(bands), valid, transform


def window(transform, x0, y0, size):
    col = int(round((x0 - transform.c) / transform.a))
    row = int(round((transform.f - y0) / -transform.e))
    return row, col


def phase_shift(a, b):
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return dy, dx, resp


def main():
    cfg = load_config()
    P = cfg["paths"]
    cal = CrossCal.load(P["processed"] / "liss4_crosscal.json")
    scenes = {str(Scene(z).date): Scene(z) for z in sorted((P["raw"] / "liss4_bhoonidhi").glob("*.zip"))}
    main_dates = sorted(d for d, s in scenes.items() if s.subscene == "B")
    extra = [d for d in scenes if d not in main_dates]
    print("[liss4] main footprint:", main_dates, "| partial overlap:", extra)

    bounds = []
    for d in main_dates:
        with rasterio.open(scenes[d].band_path(3)) as s:
            bounds.append(s.bounds)
    L, B = max(b.left for b in bounds), max(b.bottom for b in bounds)
    R, T = min(b.right for b in bounds), min(b.top for b in bounds)
    xs = np.arange(L + MARGIN * 5, R - (CHIP + MARGIN) * 5, CHIP * 5)
    ys = np.arange(T - MARGIN * 5, B + (CHIP + MARGIN) * 5, -CHIP * 5)
    cand = [(x, y) for y in ys for x in xs]
    print(f"[liss4] common footprint {(R - L) / 1000:.0f} x {(T - B) / 1000:.0f} km, {len(cand)} candidate chips")

    chips, valid_all = {}, {}
    for d in main_dates + extra:
        arr, valid, tr = read_scene(scenes[d], cal)
        H, W = valid.shape
        for k, (x, y) in enumerate(cand):
            r, c = window(tr, x - MARGIN * 5, y + MARGIN * 5, CHIP + 2 * MARGIN)
            s = CHIP + 2 * MARGIN
            if r < 0 or c < 0 or r + s > H or c + s > W:
                continue
            v = valid[r:r + s, c:c + s]
            if v.all():
                chips[(d, k)] = arr[:, r:r + s, c:c + s].copy()
                valid_all.setdefault(k, set()).add(d)
        del arr, valid
        print(f"  read {d} ({scenes[d].satellite}, sun {scenes[d].sun_elev:.0f} deg): "
              f"{sum(1 for (dd, _) in chips if dd == d)} valid chips")

    full = [k for k, ds in valid_all.items() if set(main_dates) <= ds]
    print(f"[liss4] chips valid on all {len(main_dates)} main dates: {len(full)}")

    masks, frac = {}, {}
    for (d, k), a in chips.items():
        m = detect_auto(a[:, MARGIN:-MARGIN, MARGIN:-MARGIN].astype(np.float32), "deep", 5.0)[0]
        masks[(d, k)], frac[(d, k)] = m, float((m > 0).mean())

    shifts = {REF_DATE: (0.0, 0.0)}
    for d in main_dates + extra:
        if d == REF_DATE:
            continue
        est = []
        for k in full + [k for k in valid_all if d in valid_all[k] and REF_DATE in valid_all[k]]:
            if (d, k) not in chips or (REF_DATE, k) not in chips:
                continue
            if frac[(d, k)] > 0.05 or frac[(REF_DATE, k)] > 0.05:
                continue
            dy, dx, resp = phase_shift(chips[(REF_DATE, k)][2], chips[(d, k)][2])
            if resp > 0.1:
                est.append((dy, dx))
        shifts[d] = tuple(np.median(np.array(est), 0)) if len(est) >= 3 else (0.0, 0.0)
        print(f"  shift {d} vs {REF_DATE}: dy {shifts[d][0]:+.2f} px, dx {shifts[d][1]:+.2f} px  ({len(est)} clear chips)")

    def aligned(d, k):
        a = chips[(d, k)].astype(np.float32)
        dy, dx = shifts[d]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        a = np.stack([cv2.warpAffine(b, M, b.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for b in a])
        return a[:, MARGIN:-MARGIN, MARGIN:-MARGIN]

    def aligned_mask(d, k):
        m = np.pad(masks[(d, k)], MARGIN, mode="edge")
        dy, dx = shifts[d]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(m, M, m.shape[::-1], flags=cv2.INTER_NEAREST)[MARGIN:-MARGIN, MARGIN:-MARGIN]

    xcen = {k: cand[k][0] for k in full}
    order = sorted(full, key=lambda k: xcen[k])
    n = len(order)
    west, east = order[: int(0.6 * n)], order[int(0.7 * n):]
    rng = np.random.default_rng(0)
    rng.shuffle(west)
    val, train = west[:N_VAL], west[N_VAL:N_VAL + N_TRAIN]
    test = list(rng.permutation(east))[:N_TEST]
    keep = train + val + test
    split = {k: ("train" if k in train else "val" if k in val else "test") for k in keep}
    print(f"[liss4] split: {len(train)} train, {len(val)} val, {len(test)} test chips (west/east, gap between)")

    X = np.stack([np.stack([aligned(d, k) for d in main_dates]) for k in keep]).astype(np.float16)
    Mk = np.stack([np.stack([aligned_mask(d, k) for d in main_dates]) for k in keep]).astype(np.uint8)
    F = np.array([[frac[(d, k)] for d in main_dates] for k in keep], np.float32)
    ex = {d: [k for k in keep if (d, k) in chips] for d in extra}
    extra_arr = {d: np.stack([aligned(d, k) for k in ks]).astype(np.float16) if ks else np.zeros((0, 3, CHIP, CHIP), np.float16)
                 for d, ks in ex.items()}
    extra_mask = {d: np.stack([aligned_mask(d, k) for k in ks]) if ks else np.zeros((0, CHIP, CHIP), np.uint8)
                  for d, ks in ex.items()}
    np.savez_compressed(P["processed"] / "liss4_chips.npz", X=X, mask=Mk, cloud_frac=F, dates=np.array(main_dates),
                        split=np.array([split[k] for k in keep]), xy=np.array([cand[k] for k in keep]),
                        **{f"extra_{d}": extra_arr[d] for d in extra}, **{f"extramask_{d}": extra_mask[d] for d in extra},
                        **{f"extraidx_{d}": np.array([keep.index(k) for k in ex[d]], int) for d in extra},
                        shifts=json.dumps(shifts), crs=str(scenes[main_dates[0]].profile["crs"]))
    print("[liss4] cloud fraction per date (median over kept chips):",
          {d: round(float(np.median(F[:, i])), 3) for i, d in enumerate(main_dates)})
    for d in extra:
        print(f"[liss4] {d}: {len(ex[d])} kept chips inside its footprint "
              f"({sum(split[k] == 'test' for k in ex[d])} test, {sum(split[k] == 'train' for k in ex[d])} train)")
    print("-> data/processed/liss4_chips.npz")


if __name__ == "__main__":
    main()
