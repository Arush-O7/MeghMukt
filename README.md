# MeghMukt

Cloud removal for ISRO Resourcesat LISS-IV images of North-East India, using a conditional GAN.

"Megh mukt" means "free of clouds". Most LISS-IV passes over Assam, Meghalaya and the other
north-eastern states have some cloud or cloud shadow on them, especially during the monsoon, so a
lot of scenes end up unusable. This project detects the clouds and shadows and rebuilds the ground
under them from an older clear image of the same place.

This was our B.Tech Project-I (BCSE497J) at VIT Vellore, Fall Semester 2026-27.

## What it does

1. Reads a LISS-IV scene (straight from the Bhoonidhi zip) or a Sentinel-2 image and converts it to
   reflectance. Only the Green, Red and NIR bands are used since that is what LISS-IV has.
2. Finds cloud and shadow pixels with OmniCloudMask (a rule-based detector is there as a fallback).
3. Takes an older clear image of the same area as a reference and colour-matches it.
4. Fills in the cloudy pixels with one of four methods. Clear pixels are never changed.
5. Gives a confidence map and writes a GeoTIFF that opens in QGIS/ArcGIS.

The four methods we compared:

| | Method | Notes |
|---|---|---|
| M0 | Temporal fill | no learning, copies the colour-matched reference |
| M1 | U-Net | baseline from Review 1 |
| M2 | **MeghGAN** | attention U-Net generator + PatchGAN discriminator, starts from the reference and learns the correction |
| M3 | Conditional diffusion | DDIM sampling, its sample spread is used for the confidence map |

## Results

Real clouds on 13 LISS-IV images of Guwahati (11 May 2024), checked against the clear pass of
23 May 2024. This area was not used in training. Scores are only on the cloudy pixels.

| Model | PSNR (dB) | SSIM | SAM (°) |
|---|---|---|---|
| Cloudy input | 27.42 | 0.778 | 3.97 |
| Temporal fill | 30.58 | 0.803 | 5.28 |
| U-Net | 26.61 | 0.678 | 6.67 |
| Diffusion | 31.24 | 0.798 | 4.98 |
| MeghGAN | **33.54** | **0.860** | **3.81** |

On simulated clouds over Sentinel-2 test areas, using the reference adds about 4.4 dB for MeghGAN.
Full tables are in `results/`, and the plots in `results/figures/`.

### Later update: one optimised model

After the review we looked at why results were worse on images the models had not seen. Three things
helped: combining the deep cloud mask with the rule-based one (the deep mask alone missed a lot of
cloud on some images), matching the reference colours locally around each cloud instead of once for
the whole image, and training with slightly wrong masks and shifted references. The new model
(`checkpoints/meghgan_opt.pt`, the average of the `meghgan_liss4` weights and the weights after
`--unified` training) is now used for both LISS-IV and Sentinel-2.

| PSNR on cloudy area (dB) | LISS-IV real | LISS-IV simulated | Sentinel-2 real | Sentinel-2 simulated |
|---|---|---|---|---|
| MeghGAN before | 33.54 | 33.34 | 32.25 | 36.28 |
| MeghGAN after | 34.22 | 33.59 | 32.57 | 36.43 |

Details in `results/optimisation_summary.md`.

## Running it

Python 3.10+ should work (we used 3.13 on a MacBook Air M4).

```
pip install -r requirements.txt
streamlit run app.py
```

or just double-click `run_demo.command` on macOS (`run_demo.bat` on Windows). The first run
downloads the OmniCloudMask weights, so it needs internet.

In the dashboard you can:
- search any place in North-East India in the live Sentinel-2 archive and clean a cloudy pass,
- upload your own GeoTIFF or even an ordinary JPG/PNG picture (you can also add your own reference
  image, and the original clear image if you have it, to get PSNR/SSIM/SAM for the result),
- run all four models on the same image and compare them,
- look at the evaluation tables and figures.

Tests: `./run_tests.sh` (32 tests).

## Data

The data is not in this repo because it is around 7 GB, and the LISS-IV scenes have to be
downloaded from Bhoonidhi with your own account anyway.

- **Sentinel-2 L2A**: downloaded from the Earth Search STAC catalogue by `scripts/01_download_sentinel2.py`
  and `scripts/08_download_real_pairs.py`. Ten areas, split by location so that test areas never
  show up in training (see `config.yaml`).
- **LISS-IV**: six Resourcesat-2/2A Mx scenes, path 110 row 53 (Guwahati), Nov 2023 to Nov 2024,
  from [Bhoonidhi](https://bhoonidhi.nrsc.gov.in). Put the zips in `data/raw/liss4_bhoonidhi/`.

The trained models used by the dashboard are in `checkpoints/`, so the live search and the upload
mode work without downloading anything. The built-in LISS-IV examples need the processed data.

## Training from scratch

The scripts are numbered in the order we ran them:

```
python scripts/01_download_sentinel2.py
python scripts/02_prepare_data.py
bash scripts/train_all.sh
python scripts/04_evaluate.py
python scripts/08_download_real_pairs.py
python scripts/12_prepare_real_pairs.py
bash scripts/train_real.sh
python scripts/13_prepare_liss4.py
python scripts/14_prepare_liss4_training.py
python scripts/03_train.py --model meghgan --liss4 checkpoints/meghgan_real.pt
python scripts/03_train.py --model meghgan --unified checkpoints/meghgan_liss4.pt
python scripts/15_eval_liss4.py
python scripts/07_make_figures.py
```

All the settings (areas, patch size, loss weights, training steps) are in `config.yaml`.
Each model takes roughly 10-55 minutes on an Apple M4 GPU.

## Layout

```
app.py              dashboard (Streamlit)
config.yaml         settings
meghmukt/           the package: reading data, masks, models, losses, inference, metrics
scripts/            download, preparation, training and evaluation
tests/              unit tests
checkpoints/        trained models
results/            tables, logs and figures
```

## Limitations

- With thick cloud and no reference image, any method can only guess what is underneath.
- The LISS-IV test set is one area (Guwahati) and 13 images; more areas would make the numbers
  more reliable.
- The land-cover agreement metric uses a simple k-means classification and is quite sensitive to
  small colour shifts.

## Data credits

Sentinel-2 data: contains modified Copernicus Sentinel data (2023-2025).
LISS-IV data: ISRO / NRSC, Bhoonidhi.
