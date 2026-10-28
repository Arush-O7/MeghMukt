import base64
import datetime as dt
import io
import json
import tempfile
import time
from pathlib import Path

import altair as alt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rasterio.transform import from_origin

from meghmukt.dataset import Normaliser
from meghmukt.inference import export_analysis_ready, reconstruct_scene
from meghmukt.io import Harmoniser, display_limits, get_device, load_config, read_raster, reproject_to, to_display
from meghmukt.live import (PLACES, chip_cloud, chip_profile, geocode, locate_picture, read_chip, reference_and_check,
                           search_cloudy)
from meghmukt.masking import coverage, detect_auto, pixel_size
from meghmukt.metrics import evaluate
from meghmukt.models import CONFIDENCE_MODEL, DEFAULT_MODEL, DEFAULT_MODEL_S2, ClassicalFill, default_model, load_checkpoint
from meghmukt.reference import fetch_sentinel2_reference, to_liss4_scale
from meghmukt.synth import CloudSynthesiser
from meghmukt.webimage import load_calibration, load_web_image, to_picture, visible_cloud_mask

st.set_page_config(page_title="MeghMukt", page_icon="☁️", layout="wide", initial_sidebar_state="expanded")
cfg = load_config()
P = cfg["paths"]
DEVICE = get_device(cfg["train"]["device"])

NAVY, TEAL, SLATE, AMBER = "#0B1F33", "#0E7490", "#94A3B8", "#D97706"
st.html("""<style>
:root{--ink:#0F172A;--muted:#64748B;--line:#E2E8F0;--teal:#0E7490;--navy:#0B1F33}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1400px}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.top{background:var(--navy);border-radius:8px;padding:16px 22px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.top .word{font-size:1.55rem;font-weight:700;color:#fff;letter-spacing:-.01em}
.top .div{width:1px;height:28px;background:#2C4A66}
.top .tag{color:#B8CCDC;font-size:.95rem}
.top .right{margin-left:auto;color:#7F9DB5;font-size:.8rem;text-align:right;line-height:1.4}
.meta{font-size:.84rem;color:var(--muted);margin:12px 2px 0}
.meta b{color:var(--ink);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0 2px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi .l{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600}
.kpi .v{font-size:1.4rem;font-weight:700;color:var(--ink);margin-top:2px;font-variant-numeric:tabular-nums;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi .d{font-size:.76rem;margin-top:1px;color:var(--muted)}
.kpi .d.good{color:#047857;font-weight:600}.kpi .d.bad{color:#B91C1C;font-weight:600}
.sec{margin:22px 0 6px;border-bottom:1px solid var(--line);padding-bottom:6px;display:flex;align-items:baseline;gap:12px}
.sec .k{font-size:.7rem;font-weight:700;letter-spacing:.12em;color:var(--teal);text-transform:uppercase}
.sec h3{margin:0;padding:0;font-size:1.1rem;color:var(--ink);font-weight:700}
.sec .s{color:var(--muted);font-size:.83rem;margin-left:auto}
.cap{font-size:.8rem;font-weight:600;color:#334155;display:flex;justify-content:space-between;gap:8px}
.cap span{font-weight:500;color:var(--muted);text-align:right}
.best{display:inline-block;margin-left:6px;padding:1px 7px;border-radius:4px;background:#DCFCE7;color:#065F46;
  font-size:.68rem;font-weight:700;letter-spacing:.04em}
.legend{display:flex;gap:14px;font-size:.74rem;color:var(--muted);margin-top:2px;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.ramp{height:7px;border-radius:2px;margin-top:4px}
.empty{border:1px dashed #CBD5E1;border-radius:8px;padding:24px;text-align:center;color:var(--muted);background:#fff;
  font-size:.9rem}
.empty b{color:var(--ink)}
.note{border-radius:8px;padding:10px 14px;font-size:.85rem;line-height:1.5;margin:10px 0 0}
.note.warn{background:#FEF3C7;border:1px solid #FCD34D;color:#78350F}
.note.info{background:#E0F2FE;border:1px solid #7DD3FC;color:#0C4A6E}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:42px;padding:0 18px;font-weight:600}
.sb-word{font-size:1.2rem;font-weight:700;color:#fff;padding:2px 0 0}
.sb-sub{font-size:.72rem;color:#94B4CC;padding-bottom:10px;border-bottom:1px solid #24496B;margin-bottom:4px}
.sb-step{margin:16px 0 2px;font-size:.7rem;font-weight:700;letter-spacing:.12em;color:#7DD3FC;text-transform:uppercase}
.sb-title{font-size:.92rem;font-weight:700;color:#F1F5F9;margin:0 0 2px}
.sb-note{background:#14324D;border:1px solid #24496B;border-radius:6px;padding:9px 11px;font-size:.76rem;
  color:#CFE3F2;line-height:1.45}
</style>""")


def kpi_grid(items):
    st.html('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{l}</div><div class="v" title="{v}">{v}</div><div class="d {t}">{n}</div></div>'
        for l, v, n, t in items) + "</div>")


def section(key, title, sub=""):
    st.html(f'<div class="sec"><span class="k">{key}</span><h3>{title}</h3><span class="s">{sub}</span></div>')


def panel(col, rgb, title, note="", extra="", stack=False):
    with col.container(border=True):
        if stack:
            st.html(f'<div class="cap" style="display:block"><div>{title}</div>'
                    f'<span style="display:block;text-align:left;margin-top:2px">{note}</span></div>')
        else:
            st.html(f'<div class="cap"><div>{title}</div><span>{note}</span></div>')
        st.image(rgb, width="stretch")
        if extra:
            st.html(extra)


def ramp(cmap, left, right):
    stops = ",".join(f"rgb({r * 255:.0f},{g * 255:.0f},{b * 255:.0f})" for r, g, b, _ in cmap(np.linspace(0, 1, 7)))
    return (f'<div class="ramp" style="background:linear-gradient(90deg,{stops})"></div>'
            f'<div class="legend" style="justify-content:space-between"><span>{left}</span><span>{right}</span></div>')


def step(n, title):
    st.sidebar.html(f'<div class="sb-step">Step {n}</div><div class="sb-title">{title}</div>')


def side_note(text):
    st.sidebar.html(f'<div class="sb-note">{text}</div>')


def empty(text):
    st.html(f'<div class="empty" style="margin-top:16px">{text}</div>')
    st.stop()


def b64png(rgb, max_side=900):
    im = Image.fromarray(rgb)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def before_after(before, after, height=480):
    h, w = before.shape[:2]
    st.iframe(f"""
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
.box{{position:relative;width:min(100%,{height * w / h:.0f}px);aspect-ratio:{w}/{h};margin:0 auto;border-radius:6px;
  overflow:hidden;background:#0B1F33;user-select:none}}
.box img{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
#b{{clip-path:inset(0 50% 0 0)}}
.bar{{position:absolute;top:0;bottom:0;left:50%;width:2px;margin-left:-1px;background:#fff;pointer-events:none}}
.knob{{position:absolute;top:50%;left:50%;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;
  background:#fff;box-shadow:0 1px 6px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;
  font-size:14px;color:#0B1F33;pointer-events:none}}
.tag{{position:absolute;top:8px;padding:2px 8px;border-radius:4px;background:rgba(11,31,51,.8);color:#fff;
  font-size:11px;font-weight:600;letter-spacing:.03em;pointer-events:none}}
.l{{left:8px}}.r{{right:8px}}
input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
</style>
<div class="box"><img src="data:image/png;base64,{b64png(after)}"><img id="b" src="data:image/png;base64,{b64png(before)}">
<div class="bar" id="bar"></div><div class="knob" id="k">&#8596;</div>
<span class="tag l">CLOUDY</span><span class="tag r">RECONSTRUCTED</span>
<input type="range" min="0" max="100" value="50" step="0.1" id="s" aria-label="Before/after position"></div>
<script>
const s=document.getElementById('s'),b=document.getElementById('b'),bar=document.getElementById('bar'),k=document.getElementById('k');
s.addEventListener('input',()=>{{const v=s.value;b.style.clipPath=`inset(0 ${{100-v}}% 0 0)`;bar.style.left=v+'%';k.style.left=v+'%';}});
</script>""", height=height + 6)


@st.cache_resource
def resources():
    norm = Normaliser.load(P["processed"] / "norm.json")
    harm = Harmoniser.load(P["processed"] / "harmoniser.json")
    models = {"classical": ClassicalFill()}
    for tag in (DEFAULT_MODEL, DEFAULT_MODEL_S2, "unet", CONFIDENCE_MODEL):
        if (P["checkpoints"] / f"{tag}.pt").exists():
            models[tag] = load_checkpoint(P["checkpoints"] / f"{tag}.pt", DEVICE)[0]
    return norm, harm, models, np.load(P["processed"] / "cloud_spectra.npy")


norm, harm, MODELS, SPECTRA = resources()
WEB_CAL = load_calibration(P["processed"] / "web_image.json") if (P["processed"] / "web_image.json").exists() else None
DUMMY_PROFILE = dict(driver="GTiff", crs="EPSG:32646", transform=from_origin(0, 0, 5.0, 5.0))

MODEL_INFO = {
    "MeghGAN — Conditional GAN (proposed)": dict(
        short="MeghGAN", tag=None,
        desc="<b>Proposed model.</b> Attention U-Net generator trained against a PatchGAN discriminator, with "
             "pixel, structure and spectral losses. It starts from the reference image and learns the correction."),
    "U-Net — CNN baseline": dict(
        short="U-Net", tag="unet",
        desc="Standard encoder–decoder convolutional network trained with pixel and structure losses. "
             "Baseline model from the first review."),
    "Conditional Diffusion — DDIM": dict(
        short="Diffusion", tag=CONFIDENCE_MODEL,
        desc="Generates the hidden area by removing noise step by step (20 DDIM steps). "
             "The spread of its samples also gives the confidence map."),
    "Temporal Fill — classical, non-AI": dict(
        short="Temporal Fill", tag="classical",
        desc="No learning. Colour-matches the reference image to the cloudy one and copies it into the "
             "cloudy area. The benchmark the AI models must beat."),
}


@st.cache_resource
def examples():
    out = {}
    have = lambda name: (P["processed"] / name).exists()
    tp = np.load(P["processed"] / "liss4_test_pairs.npz") if have("liss4_test_pairs.npz") else None
    ch = np.load(P["processed"] / "liss4_chips.npz") if have("liss4_chips.npz") else None
    xy, crs = (ch["xy"], str(ch["crs"])) if ch is not None else (None, None)
    for rank, i in enumerate(np.argsort(-tp["cloud"])[:3] if ch is not None and tp is not None else [], 1):
        k = int(str(tp["names"][i]).split()[0].replace("chip", ""))
        out[f"Guwahati — LISS-IV, real cloud ({rank})"] = dict(
            kind="pair", img=tp["cloudy"][i], ref=tp["ref"][i], gt=tp["truth"][i], valid=tp["valid"][i],
            profile=dict(driver="GTiff", crs=crs, transform=from_origin(float(xy[k][0]), float(xy[k][1]), 5.0, 5.0),
                         width=768, height=768),
            meta="<b>Guwahati, Assam</b> · 11 May 2024 · Resourcesat-2 LISS-IV · 5 m · ISRO Bhoonidhi",
            note="Real LISS-IV image with real cloud. Reference: clear pass of 13 Nov 2023. "
                 "Checked against the clear pass of 23 May 2024.")
    tc = np.load(P["processed"] / "liss4_test_clear.npz") if have("liss4_test_clear.npz") else None
    if tc is not None:
        out["Guwahati — LISS-IV, simulated cloud"] = dict(
            kind="synthetic", stack=tc[tc.files[0]],
            meta="<b>Guwahati, Assam</b> · Resourcesat LISS-IV · 5 m · simulated cloud on a clear image",
            note="A clear LISS-IV image with a simulated cloud, so the true ground under the cloud is known exactly.")
    states = {"agartala": "Tripura", "imphal": "Manipur"}
    for a in [a for a in cfg["split"]["test"] if have(f"s2_{a}.npz")]:
        d = np.load(P["processed"] / f"s2_{a}.npz")
        out[f"{a.title()} — Sentinel-2, real cloud"] = dict(
            kind="s2", img=d["cloudy"][0], ref=d["clear"][0],
            meta=f"<b>{a.title()}, {states.get(a, '')}</b> · {d['dates_cloudy'][0]} · Sentinel-2 L2A · 10 m",
            note=f"Real cloud. Reference: clear Sentinel-2 pass of {d['dates_clear'][0]}.")
    pic = Path("data/external/web_examples/guwahati_2025-10-12_natural_colour.jpg")
    if pic.exists():
        out["Ordinary photo (JPG)"] = dict(kind="picture", path=str(pic), meta="<b>Ordinary JPG picture</b> · no map position",
                                           note="An ordinary JPG picture, like one found on the internet.")
    return out


@st.cache_data(show_spinner="Looking up the place ...", ttl=3600)
def _geocode(q):
    return geocode(q)


@st.cache_data(show_spinner="Searching the satellite archive ...", ttl=3600)
def _search(lat, lon, start, end):
    out = []
    for r in search_cloudy(lat, lon, start, end, 5, 85, 40)[:14]:
        try:
            cf, nd = chip_cloud(r["item"], chip_profile(r["item"], lat, lon))
        except Exception:
            continue
        if 0.05 <= cf <= 0.85 and nd < 0.1:
            out.append(dict(r, chip=cf))
    return out


@st.cache_data(show_spinner="Downloading the cloudy image, a clear reference and a check image (about a minute) ...",
               ttl=3600)
def _live_load(item, lat, lon):
    prof = chip_profile(item, lat, lon)
    arr, _ = read_chip(item, prof)
    ref, ref_info, chk, chk_info = reference_and_check(prof, harm, item["properties"]["datetime"][:10])
    return harm(arr).astype(np.float32), prof, ref, ref_info, chk, chk_info


@st.cache_data(show_spinner="Finding the picture on the map (up to a minute) ...", ttl=3600)
def _locate(key, _rgb8, layout, lat, lon, km, _cloud):
    return locate_picture(_rgb8, layout, lat, lon, km, _cloud)


@st.cache_data(show_spinner="Fetching a clear reference image ...", ttl=3600)
def _fetch(key, _profile):
    return fetch_sentinel2_reference(_profile, harm, log=lambda *a: None)


@st.cache_data(show_spinner="Detecting clouds ...")
def _mask(key, _img, px):
    return detect_auto(_img, "hybrid", px)[0]


def pick_place(key, default="Shillong"):
    how = st.sidebar.radio("Place", ["From list", "Type a name"], horizontal=True, key=f"{key}_how")
    if how == "From list":
        place = st.sidebar.selectbox("Place in North-East India", list(PLACES), key=f"{key}_pick")
        return *PLACES[place], place
    q = st.sidebar.text_input("Place name", default, key=f"{key}_q")
    hit = _geocode(q) if q.strip() else None
    if hit is None:
        empty("<b>Place not found.</b><br>Try another spelling or pick from the list.")
    return hit[0], hit[1], hit[2].split(",")[0]


st.html('<div class="top"><div class="word">MeghMukt</div><div class="div"></div>'
        '<div class="tag">Generative-AI cloud removal for LISS-IV satellite imagery</div>'
        '<div class="right">North-East India<br>Green · Red · Near-infrared</div></div>')

st.sidebar.html('<div class="sb-word">MeghMukt</div><div class="sb-sub">Cloud removal</div>')
step(1, "Choose an image")
src = st.sidebar.radio("Source", ["Example images", "Search any place (live)", "Upload my image"],
                       label_visibility="collapsed")
img = gt = ref = profile = valid = None
sensor, georef, ref_text, gt_text, banner, meta = "liss4", False, "", "", "", ""
pic_bytes, pic_layout, pic_rgb, pic_info = None, "natural", None, None

if src == "Example images":
    E = examples()
    e = E[st.sidebar.selectbox("Example", list(E))]
    side_note(e["note"])
    meta = e["meta"]
    if e["kind"] == "pair":
        img, ref, gt = (e[k].astype(np.float32) for k in ("img", "ref", "gt"))
        valid, profile, georef = e["valid"], e["profile"], True
        ref_text, gt_text = "clear pass, 13 Nov 2023", "clear pass, 23 May 2024"
    elif e["kind"] == "synthetic":
        gt, ref = e["stack"][0].astype(np.float32), e["stack"][1].astype(np.float32)
        img = CloudSynthesiser(SPECTRA, (0.4, 0.401))(gt, np.random.default_rng(3))[0]
        ref_text, gt_text = "clear LISS-IV pass", "original image before the simulated cloud"
    elif e["kind"] == "s2":
        img, ref = e["img"].astype(np.float32), e["ref"].astype(np.float32)
        sensor, ref_text = "s2", "clear Sentinel-2 pass"
    else:
        pic_bytes, sensor = Path(e["path"]).read_bytes(), "s2"

elif src == "Search any place (live)":
    sensor = "s2"
    lat, lon, place = pick_place("live")
    today = dt.date.today()
    try:
        passes = _search(round(lat, 4), round(lon, 4), str(today - dt.timedelta(days=365)), str(today))
    except Exception as e:
        empty(f"<b>Could not reach the satellite archive.</b><br>Check the internet connection. ({type(e).__name__})")
    if not passes:
        empty("<b>No partly cloudy image found here in the last year.</b><br>Try another place.")
    k = st.sidebar.selectbox("Cloudy image (date · cloud over the area)", range(len(passes)),
                             index=int(np.argmin([abs(p["chip"] - 0.35) for p in passes])),
                             format_func=lambda i: f"{passes[i]['date']} · {passes[i]['chip']:.0%} cloud")
    try:
        img, profile, ref, ref_info, chk, chk_info = _live_load(passes[k]["item"], round(lat, 4), round(lon, 4))
    except Exception as e:
        empty(f"<b>Download failed.</b><br>Try again or pick another date. ({type(e).__name__})")
    georef = True
    meta = f"<b>{place}</b> · {passes[k]['date']} · Sentinel-2 L2A · 10 m · downloaded live"
    if ref is not None:
        ref_text = f"clear pass, {ref_info['reference_date']}"
    if chk is not None:
        gt, gt_text = chk, f"clear pass, {chk_info['reference_date']}"
    banner = (f'<div class="note info"><b>Live data.</b> Sentinel-2 image of {passes[k]["date"]} over {place}, '
              'downloaded from the public Copernicus archive.'
              + (f' The clear image of {chk_info["reference_date"]} is used to check the result.' if chk is not None else "")
              + '</div>')

else:
    up = st.sidebar.file_uploader("Satellite image (GeoTIFF) or picture (PNG / JPG)",
                                  type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up is None:
        empty("<b>No image uploaded.</b><br>Upload a satellite GeoTIFF or an ordinary picture in the sidebar.")
    arr = None
    if up.name.lower().endswith((".tif", ".tiff")):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            f.write(up.getvalue())
        arr, profile = read_raster(f.name)
        if arr.dtype == np.uint8 and arr.shape[0] in (3, 4) and profile.get("crs") is None:
            arr, profile = None, None
    if arr is None:
        pic_bytes, sensor = up.getvalue(), "s2"
        if st.sidebar.toggle("Picture is in false colour (plants look red)", value=False):
            pic_layout = "fcc"
        meta = f"<b>{up.name}</b> · ordinary picture · no map position"
    else:
        is_s2 = any(s in up.name.lower() for s in ("sentinel", "s2_", "_s2"))
        sensor = st.sidebar.radio("Satellite", ["liss4", "s2"], horizontal=True, index=int(is_s2),
                                  format_func=lambda v: {"liss4": "LISS-IV", "s2": "Sentinel-2"}[v])
        bands = [3, 4, 8] if sensor == "s2" and arr.shape[0] >= 8 else [1, 2, 3]
        img = to_liss4_scale(arr[[b - 1 for b in bands]], sensor, harm)
        georef = profile.get("crs") is not None
        meta = f"<b>{up.name}</b> · {'Sentinel-2' if sensor == 's2' else 'LISS-IV'}" + (" · georeferenced" if georef else "")
        with st.sidebar.expander("Optional: own reference / original image"):
            up_ref = st.file_uploader("Reference image (older clear pass of the same place)", type=["tif", "tiff"])
            up_gt = st.file_uploader("Original clear image (to score the result)", type=["tif", "tiff"])
        for kind, extra in (("ref", up_ref), ("gt", up_gt)):
            if extra is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                f.write(extra.getvalue())
            ea, ep = read_raster(f.name)
            ea = ea[[b - 1 for b in bands]] if ea.shape[0] >= max(bands) else ea[:3]
            if georef and ep.get("crs") is not None and (ep["crs"], ep["transform"], ep["width"], ep["height"]) != (
                    profile["crs"], profile["transform"], profile["width"], profile["height"]):
                ea = np.nan_to_num(reproject_to(ea, ep, profile))
            if ea.shape[1:] != img.shape[1:]:
                side_note(f"<b>{extra.name}</b> does not cover the same area, so it is not used.")
                continue
            ea = to_liss4_scale(ea, sensor, harm)
            if kind == "ref":
                ref, ref_text = ea, f"uploaded ({extra.name})"
            else:
                gt, gt_text = ea, f"original clear image ({extra.name})"

if pic_bytes is not None:
    if WEB_CAL is None:
        empty("Run <b>scripts/16_web_image_fit.py</b> first.")
    img, pic_rgb, pic_info = load_web_image(pic_bytes, pic_layout, harm, WEB_CAL)
    banner = ('<div class="note warn"><b>Ordinary picture.</b> It has no location and no calibrated brightness, so '
              'without a reference the area under the cloud can only be estimated. Turn on <b>Find this picture on the '
              'map</b> in step 2 to rebuild it from real satellite data.</div>')

px_m = 10.0 if sensor == "s2" else pixel_size(profile, 5.0)
if pic_rgb is not None and pic_layout == "natural":
    mask = visible_cloud_mask(pic_rgb)
else:
    mask = _mask(hash(img.tobytes()), img, px_m)

step(2, "Reference image")
if pic_rgb is not None:
    if st.sidebar.toggle("Find this picture on the map", value=False,
                         help="Matches the picture to a clear satellite image of the place, which then serves as "
                              "the reference."):
        lat, lon, place = pick_place("loc")
        km = st.sidebar.select_slider("Search area (km)", [5, 10, 15, 20, 30], value=15)
        try:
            found, msg = _locate(hash(pic_bytes) ^ hash(pic_layout), pic_rgb, pic_layout, round(lat, 4), round(lon, 4),
                                 km, visible_cloud_mask(pic_rgb, buffer=0))
        except Exception as e:
            found, msg = None, f"the archive could not be reached ({type(e).__name__})"
        if found is None:
            side_note(f"<b>Not found:</b> {msg}. Check the place or change the search area.")
        else:
            ref, profile, georef = harm(found["refl"]), found["profile"], True
            ref_text = f"clear pass, {found['date']}"
            side_note(f"<b>Found</b> near {place} ({found['inliers']} matching features). "
                      f"Reference: clear satellite image of {found['date']}.")
            banner = (f'<div class="note info"><b>Picture located</b> near {place} by matching {found["inliers"]} '
                      f'ground features with a clear satellite image of {found["date"]}. That image is used as the '
                      'reference, so the area under the cloud is rebuilt from real data.</div>')
    else:
        side_note("No reference: the area under the cloud will only be estimated.")
elif ref is not None:
    if not st.sidebar.toggle("Use the reference image", value=True,
                             help="An older, clear image of the same place. Switch off to see the result without it."):
        ref, ref_text = None, ""
    side_note(f"Reference: {ref_text}." if ref is not None else "No reference: the area under the cloud will only be "
                                                                "estimated.")
elif georef:
    if st.sidebar.toggle("Fetch a clear reference automatically", value=True):
        ref, info = _fetch(str(profile["transform"]) + str(profile["crs"]), profile)
        ref_text = f"clear pass, {info['reference_date']}" if ref is not None else ""
        side_note(f"Reference: {ref_text}." if ref is not None else "No clear image found for this place.")
else:
    side_note("This image has no map position, so no reference can be fetched.")

step(3, "Model")
choice = st.sidebar.selectbox("Model", list(MODEL_INFO), label_visibility="collapsed")
side_note(MODEL_INFO[choice]["desc"])
compare = st.sidebar.toggle("Compare all four models", value=False,
                            help="Runs every model on the same image and ranks them.")
view = st.sidebar.radio("Colours", ["fcc", "natural"], horizontal=True,
                        format_func=lambda v: {"natural": "Natural colour", "fcc": "False colour"}[v],
                        help="False colour shows near-infrared as red, so healthy vegetation looks red. "
                             "It is the standard view for LISS-IV.")


def tag_of(name):
    return MODEL_INFO[name]["tag"] or default_model("s2" if sensor == "s2" else "liss4")


tab_run, tab_res = st.tabs(["Cloud removal", "Evaluation"])

with tab_run:
    cov = coverage(mask)
    H, W = img.shape[1:]
    if meta:
        st.html(f'<div class="meta">{meta}</div>')
    kpi_grid([("Image size", f"{W}×{H} px", f"{W * px_m / 1000:.1f} × {H * px_m / 1000:.1f} km ground", ""),
              ("Cloud cover", f"{cov['cloud'] + cov['shadow']:.0%}",
               f"cloud {cov['cloud']:.0%} · shadow {cov['shadow']:.0%}", ""),
              ("Reference image", "Yes" if ref is not None else "No", ref_text or "estimate only", ""),
              ("Model", "All four" if compare else MODEL_INFO[choice]["short"],
               "comparison run" if compare else choice.split("— ")[1], "")])
    if banner:
        st.html(banner)

    section("Step 1", "Input", "Detected clouds are the only pixels that will change")
    lim = display_limits(gt if gt is not None else (ref if ref is not None else img), view)
    shown_in = pic_rgb if pic_rgb is not None else to_display(img, view, display_limits(img, view))
    overlay = shown_in.copy()
    overlay[mask == 1] = (0.45 * overlay[mask == 1] + np.array([20, 120, 230]) * 0.55).astype(np.uint8)
    overlay[mask == 2] = (0.45 * overlay[mask == 2] + np.array([245, 150, 30]) * 0.55).astype(np.uint8)
    cols = st.columns(3 if ref is not None else 2)
    panel(cols[0], shown_in, "Cloudy image")
    panel(cols[1], overlay, "Detected clouds", f"{cov['cloud'] + cov['shadow']:.0%} of the image",
          '<div class="legend"><span><i style="background:#1478E6"></i>Cloud</span>'
          '<span><i style="background:#F5961E"></i>Cloud shadow</span></div>')
    if ref is not None:
        panel(cols[2], to_display(ref, view, display_limits(ref, view)), "Reference image", ref_text)

    section("Step 2", "Reconstruction", "Runs on the Apple GPU" if DEVICE.type == "mps" else "")
    names = list(MODEL_INFO) if compare else [choice]
    run_key = hash((img.tobytes(), mask.tobytes(), None if ref is None else ref.tobytes(), tuple(names), choice))
    if st.button("Run comparison of all four models" if compare else f"Remove clouds with {MODEL_INFO[choice]['short']}",
                 type="primary", width="stretch"):
        results = {}
        for name in names:
            bar = st.progress(0.0, text=f"Running {MODEL_INFO[name]['short']} ...")
            t0 = time.perf_counter()
            out, _, conf = reconstruct_scene(
                MODELS[tag_of(name)], img, mask, ref, norm, DEVICE, cfg["inference"]["tile"], cfg["inference"]["overlap"],
                tta=True, conf_model=MODELS.get(CONFIDENCE_MODEL) if name == choice else None,
                progress=lambda f, b=bar, n=name: b.progress(f, text=f"Running {MODEL_INFO[n]['short']} ..."))
            bar.empty()
            results[name] = (out, conf, time.perf_counter() - t0)
        st.session_state["result"] = (run_key, results)

    saved = st.session_state.get("result")
    if saved is None or saved[0] != run_key:
        st.html('<div class="empty">Press the button above to run the reconstruction.</div>')
    else:
        results = saved[1]
        region = mask > 0
        scored = None if gt is None else (region if valid is None else region & valid)
        base = evaluate(img, gt, scored) if gt is not None else None
        shown = {n: (to_picture(o, mask, pic_rgb, pic_info, harm) if pic_rgb is not None else to_display(o, view, lim))
                 for n, (o, _, _) in results.items()}
        scores = {n: evaluate(o, gt, scored) for n, (o, _, _) in results.items()} if gt is not None else {}

        if compare:
            section("Step 3", "Model comparison", "Same image, same mask, same reference")
            best = max(scores, key=lambda n: scores[n]["psnr_cloud"]) if scores else None
            gcols = st.columns(4)
            for c, n in zip(gcols, names):
                sc = scores.get(n)
                note = (f"PSNR {sc['psnr_cloud']:.1f} dB · SSIM {sc['ssim_cloud']:.2f}" if sc
                        else f"{results[n][2]:.0f} s")
                panel(c, shown[n], MODEL_INFO[n]["short"] + ('<span class="best">BEST</span>' if n == best else ""), note,
                      stack=True)
            if scores:
                tbl = pd.DataFrame([{"Model": MODEL_INFO[n]["short"], "PSNR (dB)": s["psnr_cloud"], "SSIM": s["ssim_cloud"],
                                     "SAM (°)": s["sam_cloud"], "Mean abs. error": s["mae_cloud"],
                                     "Time (s)": results[n][2]} for n, s in scores.items()])
                c1, c2 = st.columns([3, 2], gap="medium")
                with c1, st.container(border=True):
                    st.html(f'<div class="cap"><div>Scores on this image</div><span>compared with the truth '
                            f'({gt_text}); best in green</span></div>')
                    lower = ("SAM (°)", "Mean abs. error", "Time (s)")
                    st.dataframe(tbl.style.apply(lambda s: ["background-color:#DCFCE7;color:#065F46;font-weight:600"
                                                            if v == (s.min() if s.name in lower else s.max()) else ""
                                                            for v in s], subset=tbl.columns[1:])
                                 .format({"PSNR (dB)": "{:.2f}", "SSIM": "{:.3f}", "SAM (°)": "{:.2f}",
                                          "Mean abs. error": "{:.4f}", "Time (s)": "{:.1f}"}), hide_index=True)
                    st.caption(f"Cloudy image before reconstruction: PSNR {base['psnr_cloud']:.1f} dB, "
                               f"SSIM {base['ssim_cloud']:.2f}. PSNR and SSIM: higher is better. SAM and error: lower is better.")
                with c2, st.container(border=True):
                    st.html('<div class="cap"><div>Ranking</div><span>PSNR on the cloudy area, dB</span></div>')
                    d = tbl[["Model", "PSNR (dB)"]].copy()
                    d["Best"] = d["Model"] == MODEL_INFO[best]["short"]
                    b = alt.Chart(d).encode(
                        y=alt.Y("Model:N", sort="-x", title=None,
                                axis=alt.Axis(labelColor="#334155", ticks=False, domain=False, labelOverlap=False)),
                        x=alt.X("PSNR (dB):Q", title=None, scale=alt.Scale(domain=[0, float(d["PSNR (dB)"].max()) * 1.15]),
                                axis=alt.Axis(gridColor="#EEF2F7", labelColor="#64748B", domain=False)))
                    chart = (b.mark_bar(cornerRadiusEnd=3, height=18).encode(
                        color=alt.condition(alt.datum.Best, alt.value(TEAL), alt.value(SLATE)))
                        + b.mark_text(align="left", dx=5, fontSize=11, color="#0F172A").encode(
                            text=alt.Text("PSNR (dB):Q", format=".1f")))
                    st.altair_chart(chart.properties(height=alt.Step(34)).configure_view(stroke=None), width="stretch")

        out, conf, secs = results[choice]
        section("Step 4" if compare else "Step 3", f"Result — {MODEL_INFO[choice]['short']}", "Drag the handle to compare")
        cards = [("Area rebuilt", f"{region.mean():.0%}", "of the image", ""),
                 ("Confidence", f"{conf[region].mean() if region.any() else 1:.0%}", "average on the rebuilt area", ""),
                 ("Run time", f"{secs:.0f} s", "", "")]
        if gt is not None:
            m = scores[choice]
            dp = m["psnr_cloud"] - base["psnr_cloud"]
            cards = [("PSNR", f"{m['psnr_cloud']:.1f} dB", f"{dp:+.1f} dB vs cloudy image", "good" if dp > 0 else "bad"),
                     ("SSIM", f"{m['ssim_cloud']:.2f}", "1.00 = identical to the truth", ""),
                     ("Spectral angle", f"{m['sam_cloud']:.1f}°", "colour error, lower is better", "")] + cards
        kpi_grid(cards)
        if gt is not None:
            st.caption(f"Scores are measured on the cloudy area only, against the truth: {gt_text}.")

        left, right = st.columns([3, 2], gap="medium")
        with left, st.container(border=True):
            st.html('<div class="cap"><div>Before and after</div><span>drag the handle</span></div>')
            before_after(pic_rgb if pic_rgb is not None else to_display(img, view, lim), shown[choice])
        with right:
            tiles = [(shown[choice], "Reconstructed image", ""),
                     ((cm.RdYlGn(conf)[..., :3] * 255).astype(np.uint8), "Confidence map", "",
                      ramp(cm.RdYlGn, "low", "high"))]
            if gt is not None:
                err = np.abs(out - gt).mean(0)
                err_rgb = (cm.inferno(np.clip(err / 0.15, 0, 1))[..., :3] * 255).astype(np.uint8)
                err_rgb[~region] = 235
                tiles += [(err_rgb, "Error map", "rebuilt area", ramp(cm.inferno, "no error", "large error")),
                          (to_display(gt, view, lim) if pic_rgb is None else shown[choice], "Truth", gt_text)]
            for i in range(0, len(tiles), 2):
                row = st.columns(2)
                for c, t in zip(row, tiles[i:i + 2]):
                    panel(c, *t)

        prof = dict(profile or dict(DUMMY_PROFILE), width=W, height=H)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            export_analysis_ready(f.name, out, mask, conf, prof, choice)
            tif = Path(f.name).read_bytes()
        png = io.BytesIO()
        Image.fromarray(shown[choice]).save(png, format="PNG")
        d1, d2 = st.columns(2)
        d1.download_button("Download image (PNG)", png.getvalue(), file_name="meghmukt_result.png", mime="image/png",
                           width="stretch")
        d2.download_button("Download GeoTIFF for GIS (bands + cloud mask + confidence)", tif,
                           file_name="meghmukt_result.tif", mime="image/tiff", width="stretch")


@st.cache_data
def md_tables(path: str, mtime: float):
    tables, head, rows = [], "", []

    def flush():
        if rows:
            cols = [c.strip() for c in rows[0].strip("|").split("|")]
            df = pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]], columns=cols)
            df = df.rename(columns={cols[0]: "Model"})
            for c in df.columns[1:]:
                v = pd.to_numeric(df[c].str.rstrip("%"), errors="coerce")
                if v.notna().all():
                    df[c] = v
            tables.append((head, df))
            rows.clear()

    for line in Path(path).read_text().splitlines():
        if line.startswith("|"):
            rows.append(line)
            continue
        flush()
        if line.startswith("#"):
            head = line.lstrip("# ").strip()
    flush()
    return tables


CLOUDY = "Cloudy image (no processing)"


def simple(df, meghgan, diffusion, psnr="PSNR cloud", ssim="SSIM"):
    rows = []
    for label, name in (("Do nothing", CLOUDY), ("M0 Classical temporal fill", "Temporal Fill (classical)"),
                        ("M1 U-Net baseline", "U-Net"), (meghgan, "MeghGAN (proposed)"),
                        (diffusion, "Conditional Diffusion")):
        r = df[df["Model"] == label]
        if len(r):
            r = r.iloc[0]
            rows.append({"Model": name, "PSNR (dB)": r[psnr], "SSIM": r[ssim], "Mean abs. error": r["MAE"],
                         "SAM (°)": r["SAM °"], "NDVI error": r["NDVI MAE"], "Land-cover match (%)": r["Land-cover"]})
    return pd.DataFrame(rows)


LOWER = ("error", "SAM")


def fmt_of(c):
    return ("{:.1f}%" if "%" in c else "{:.4f}" if "Mean abs" in c else "{:.3f}" if ("SSIM" in c or "NDVI" in c)
            else "{:.2f}")


def show_table(t, title, note=""):
    real = t["Model"] != CLOUDY

    def col(s):
        if s.name == "Model":
            return ["" if r else "color:#94A3B8" for r in real]
        best = s[real].min() if any(k in s.name for k in LOWER) else s[real].max()
        return ["background-color:#DCFCE7;color:#065F46;font-weight:600" if r and v == best else
                ("" if r else "color:#94A3B8") for r, v in zip(real, s)]
    with st.container(border=True):
        st.html(f'<div class="cap"><div>{title}</div><span>{note}</span></div>')
        st.dataframe(t.style.apply(col).format({c: fmt_of(c) for c in t.columns[1:]}), hide_index=True)


def bar_chart(d, value, highlight):
    d = d.copy()
    d["Highlight"] = d["Model"].map(highlight)
    base = alt.Chart(d).encode(
        y=alt.Y("Model:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=300, labelColor="#334155", labelOverlap=False, ticks=False, domain=False)),
        x=alt.X(f"{value}:Q", title=None, scale=alt.Scale(domain=[0, float(d[value].max()) * 1.12]),
                axis=alt.Axis(gridColor="#EEF2F7", labelColor="#64748B", domain=False)),
        tooltip=["Model", alt.Tooltip(f"{value}:Q", format=".2f")])
    bars = base.mark_bar(cornerRadiusEnd=3, height=16).encode(
        color=alt.condition(alt.datum.Highlight, alt.value(TEAL), alt.value(SLATE)))
    text = base.mark_text(align="left", dx=5, fontSize=11, color="#0F172A").encode(text=alt.Text(f"{value}:Q", format=".1f"))
    st.altair_chart((bars + text).properties(height=alt.Step(30)).configure_view(stroke=None), width="stretch")


METRIC_GUIDE = ("**PSNR (dB)** — how close the rebuilt pixels are to the real ground (higher is better) · "
                "**SSIM** — similarity of shapes and texture, 0 to 1 (higher) · "
                "**Mean abs. error** — average reflectance difference (lower) · "
                "**SAM (°)** — colour / spectral error (lower) · "
                "**NDVI error** — error in the vegetation index used for crop and forest maps (lower) · "
                "**Land-cover match** — share of pixels that get the same land-cover class as the truth (higher). "
                "All scores are measured on the cloudy area only. Best value in each column is green.")

with tab_res:
    lp, rp, sm = P["results"] / "liss4_summary.md", P["results"] / "real_pairs_summary.md", P["results"] / "summary.csv"
    if not (lp.exists() and rp.exists() and sm.exists()):
        empty("No evaluation results yet. Run the evaluation scripts first.")
    jsons = {n: json.loads((P["results"] / f"{n}.json").read_text())
             for n in ("mask_benchmark", "blue_band_study", "uncertainty_study") if (P["results"] / f"{n}.json").exists()}
    l4 = {h[0].lower(): t for h, t in md_tables(str(lp), lp.stat().st_mtime)}
    l4_real = simple(l4["b"], "M2 + real LISS-IV (Bhoonidhi) training", "M3 Diffusion + real-cloud training")
    l4_sim = simple(l4["a"], "M2 + real LISS-IV (Bhoonidhi) training", "M3 Diffusion + real-cloud training")
    s2 = next(t for h, t in md_tables(str(rp), rp.stat().st_mtime) if "deep" in h)
    s2_real = simple(s2, "M2 MeghGAN + real-cloud training", "M3 Diffusion + real-cloud training", ssim="SSIM cloud")
    df = pd.read_csv(sm)

    o, t_l4, t_s2, t_syn, t_abl, t_aux, t_fig = st.tabs(["Overview", "Real LISS-IV clouds", "Real Sentinel-2 clouds",
                                                          "With vs without reference", "Design choices",
                                                          "Supporting studies", "Figures"])
    with o:
        ours = l4_real.set_index("Model").loc["MeghGAN (proposed)"]
        fill = l4_real.set_index("Model").loc["Temporal Fill (classical)"]
        g = df[(df["set"] == "test") & (df["model"] == "meghgan")].set_index("scenario")["psnr_cloud"]
        cards = [("MeghGAN on real LISS-IV clouds", f"{ours['PSNR (dB)']:.1f} dB",
                  f"{ours['PSNR (dB)'] - fill['PSNR (dB)']:+.1f} dB vs Temporal Fill", "good"),
                 ("Similarity to the truth", f"{ours['SSIM']:.2f}", "SSIM, 1.00 = identical", ""),
                 ("Gain from reference image", f"{g['with_ref'] - g['no_ref']:+.1f} dB", "MeghGAN with vs without", "good")]
        if "mask_benchmark" in jsons:
            cards.append(("Clouds detected", f"{jsons['mask_benchmark']['results']['deep']['cloud']['recall']:.0%}",
                          "by the cloud detector", ""))
        kpi_grid(cards)
        st.caption(METRIC_GUIDE)
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            st.html('<div class="cap"><div>Real LISS-IV clouds</div><span>PSNR, dB — higher is better</span></div>')
            bar_chart(l4_real[l4_real["Model"] != CLOUDY], "PSNR (dB)", lambda m: m.startswith("MeghGAN"))
            st.caption("13 real cloudy LISS-IV images from ISRO Bhoonidhi, checked against a clear image 12 days later.")
        with c2, st.container(border=True):
            st.html('<div class="cap"><div>Effect of the reference image</div><span>PSNR, dB</span></div>')
            fam = {"classical": "Temporal Fill", "unet": "U-Net", "meghgan": "MeghGAN", "diffusion": "Diffusion"}
            gd = df[(df["set"] == "test") & df["model"].isin(fam)].copy()
            gd["Model"] = gd["model"].map(fam)
            gd["Reference"] = gd["scenario"].map({"with_ref": "With reference", "no_ref": "Without reference"})
            ch = alt.Chart(gd).mark_bar(cornerRadiusEnd=3, height=11).encode(
                y=alt.Y("Model:N", title=None, sort=list(fam.values()),
                        axis=alt.Axis(labelColor="#334155", labelOverlap=False, ticks=False, domain=False)),
                yOffset=alt.YOffset("Reference:N", sort=["With reference", "Without reference"]),
                x=alt.X("psnr_cloud:Q", title=None, scale=alt.Scale(domain=[0, 42]),
                        axis=alt.Axis(gridColor="#EEF2F7", labelColor="#64748B", domain=False)),
                color=alt.Color("Reference:N", scale=alt.Scale(domain=["With reference", "Without reference"],
                                                               range=[TEAL, AMBER]),
                                legend=alt.Legend(orient="top", title=None, labelColor="#334155", direction="horizontal")),
                tooltip=["Model", "Reference", alt.Tooltip("psnr_cloud:Q", format=".1f", title="PSNR")])
            st.altair_chart(ch.properties(height=alt.Step(13)).configure_view(stroke=None), width="stretch")
            st.caption("Sentinel-2 images of Agartala and Imphal with simulated clouds.")

    with t_l4:
        st.caption("ISRO Resourcesat LISS-IV images of Guwahati from the Bhoonidhi portal. The test area (east part) was "
                   "never used for training. " + METRIC_GUIDE)
        show_table(l4_real, "Real clouds", "13 images · truth = clear pass 12 days later")
        show_table(l4_sim, "Simulated clouds", "48 images · truth = the image before the cloud was added")

    with t_s2:
        st.caption("Sentinel-2 images with real clouds over areas never used for training. Each is checked against a "
                   "clear image of the same place taken at most 10 days later. " + METRIC_GUIDE)
        show_table(s2_real, "Real clouds", "13 images · 9 places in North-East India")

    with t_syn:
        st.caption("The same models run twice: once with an older clear image of the place (the reference) and once "
                   "without it. Sentinel-2 test areas, simulated clouds. PSNR in dB, higher is better.")
        rows = []
        for code, name in (("copy_input", CLOUDY), ("classical", "Temporal Fill (classical)"), ("unet", "U-Net"),
                           ("meghgan", "MeghGAN (proposed)"), ("diffusion", "Conditional Diffusion")):
            r = df[(df["set"] == "test") & (df["model"] == code)].set_index("scenario")
            if "with_ref" in r.index:
                w = r.loc["with_ref"]
                n = r.loc["no_ref"] if "no_ref" in r.index else w
                rows.append({"Model": name, "PSNR with reference": w["psnr_cloud"], "PSNR without": n["psnr_cloud"],
                             "SSIM with reference": w["ssim_cloud"], "SSIM without": n["ssim_cloud"],
                             "SAM (°) with reference": w["sam_cloud"], "SAM (°) without": n["sam_cloud"]})
        show_table(pd.DataFrame(rows), "With and without the reference image", "48 images · simulated clouds")
        st.caption("U-Net does not use the reference, so its score is the same in both cases. The other models lose "
                   "quality without it, because the ground under thick cloud can then only be estimated. With the "
                   "reference, MeghGAN is the best model.")

    with t_abl:
        st.caption("Each row changes one thing in MeghGAN and retrains it; everything else stays the same. This shows "
                   "why each part of the model is there.")
        sel = lambda st_, code: df[(df["set"] == st_) & (df["scenario"] == "with_ref") & (df["model"] == code)]
        rows = []
        for code, name in (("meghgan", "MeghGAN — full model"), ("meghgan_nospectral", "without spectral loss (SAM + NDVI)"),
                           ("meghgan_noadv", "without GAN (adversarial) loss"),
                           ("meghgan_charb", "Charbonnier loss instead of L1")):
            a, b = sel("test", code), sel("liss4", code)
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                rows.append({"Model": name, "PSNR (dB) · Sentinel-2": a["psnr_cloud"], "SSIM · Sentinel-2": a["ssim_cloud"],
                             "PSNR (dB) · LISS-IV": b["psnr_cloud"], "SAM (°) · LISS-IV": b["sam_cloud"],
                             "NDVI error · LISS-IV": b["ndvi_mae_cloud"]})
        abl = pd.DataFrame(rows)
        show_table(abl, "Loss functions", "simulated clouds with reference · Sentinel-2 test areas and LISS-IV tiles")
        if len(abl) == 4:
            full, nospec, noadv = abl.iloc[0], abl.iloc[1], abl.iloc[2]
            st.caption(f"Removing the spectral loss costs {full['PSNR (dB) · LISS-IV'] - nospec['PSNR (dB) · LISS-IV']:.1f} dB "
                       f"on LISS-IV and raises the colour error from {full['SAM (°) · LISS-IV']:.2f}° to "
                       f"{nospec['SAM (°) · LISS-IV']:.2f}°, so it is kept. The GAN loss changes PSNR by only "
                       f"{full['PSNR (dB) · Sentinel-2'] - noadv['PSNR (dB) · Sentinel-2']:+.1f} dB on Sentinel-2 (and is "
                       f"slightly lower on LISS-IV); it is kept because it gives sharper, more natural texture, which "
                       "PSNR does not reward.")

        st.html('<div style="height:6px"></div>')
        c1, c2 = st.columns(2, gap="medium")
        b4 = l4["b"].set_index("Model")
        s2_pairs = next((m for m in b4.index if m.startswith("M2 MeghGAN + real-cloud training")), "")
        steps = [("Simulated clouds only", "M2 MeghGAN (ours)"),
                 ("+ real Sentinel-2 cloud pairs", s2_pairs),
                 ("+ real LISS-IV images (final)", "M2 + real LISS-IV (Bhoonidhi) training")]
        t1 = pd.DataFrame([{"Model": n, "PSNR (dB)": b4.loc[k, "PSNR cloud"], "SSIM": b4.loc[k, "SSIM"],
                            "SAM (°)": b4.loc[k, "SAM °"]} for n, k in steps if k in b4.index])
        s2i = s2.set_index("Model")
        t2 = pd.DataFrame([{"Model": n, "PSNR (dB)": s2i.loc[k, "PSNR cloud"], "SSIM": s2i.loc[k, "SSIM cloud"],
                            "SAM (°)": s2i.loc[k, "SAM °"]}
                           for n, k in (("Simulated clouds only", "M2 MeghGAN (ours)"),
                                        ("+ real Sentinel-2 cloud pairs (final)", "M2 MeghGAN + real-cloud training"))
                           if k in s2i.index])
        with c1:
            show_table(t1, "Training data — real LISS-IV clouds", "13 images")
        with c2:
            show_table(t2, "Training data — real Sentinel-2 clouds", "13 images")
        st.caption("Training only on simulated clouds is not enough for real clouds. Adding real cloudy/clear pairs, and "
                   "finally real ISRO LISS-IV images, improves the result step by step. This is why the dashboard uses a "
                   "different final MeghGAN for each satellite.")

    with t_aux:
        a1 = a2 = a3 = st.container()
        if "mask_benchmark" in jsons:
            mb = jsons["mask_benchmark"]["results"]
            t = pd.DataFrame([{"Model": "Deep detector (used)", "Clouds found": mb["deep"]["cloud"]["recall"],
                               "Correct flags": mb["deep"]["cloud"]["precision"]},
                              {"Model": "Physics rules", "Clouds found": mb["rules"]["cloud"]["recall"],
                               "Correct flags": mb["rules"]["cloud"]["precision"]}])
            with a1, st.container(border=True):
                st.html('<div class="cap"><div>Cloud detection</div><span>vs ESA cloud labels</span></div>')
                st.dataframe(t.rename(columns={"Model": "Detector"}).style.format({"Clouds found": "{:.0%}",
                                                                                  "Correct flags": "{:.0%}"}),
                             hide_index=True)
                st.caption("The deep detector misses fewer clouds, which matters more: a missed cloud stays in the image.")
        if "blue_band_study" in jsons:
            bb = jsons["blue_band_study"]["results"]
            t = pd.DataFrame([{"Method": "Our fitted model", "Accuracy (R²)": bb["ratio-fit"]["r2"]},
                              {"Method": "Fixed polynomial formula", "Accuracy (R²)": bb["fixed_poly"]["r2"]},
                              {"Method": "NDVI/NDWI ratio formula", "Accuracy (R²)": bb["ndvi_ndwi_ratio"]["r2"]}])
            with a2, st.container(border=True):
                st.html('<div class="cap"><div>Blue band for natural colour</div><span>R², 1.0 = perfect</span></div>')
                st.dataframe(t.style.format({"Accuracy (R²)": "{:.2f}"}), hide_index=True)
                st.caption("LISS-IV has no blue band, so it is predicted from the other three bands.")
        if "uncertainty_study" in jsons:
            us = jsons["uncertainty_study"]["results"]
            t = pd.DataFrame([{"Method": "Diffusion spread × cloud depth (used)",
                               "Agreement with real error": us["deployed: diffusion spread x depth"]["unc_spearman"]},
                              {"Method": "Flipped/rotated copies (first try)",
                               "Agreement with real error": us["MeghGAN test-time-augmentation spread"]["unc_spearman"]}])
            with a3, st.container(border=True):
                st.html('<div class="cap"><div>Confidence map</div><span>rank correlation, higher is better</span></div>')
                st.dataframe(t.style.format({"Agreement with real error": "{:.2f}"}), hide_index=True)
                st.caption("Low-confidence pixels have about twice the error of high-confidence ones.")

    with t_fig:
        figs = [("liss4_real_pairs", "Real LISS-IV cloud cases"), ("real_pairs", "Real Sentinel-2 cloud cases"),
                ("reference_ablation", "Effect of the reference image"), ("training_curves", "Training curves")]
        fc = st.columns(2, gap="medium")
        for i, (f, title) in enumerate(figs):
            p = P["results"] / "figures" / f"{f}.png"
            if p.exists():
                with fc[i % 2], st.container(border=True):
                    st.html(f'<div class="cap"><div>{title}</div></div>')
                    st.image(str(p), width="stretch")
