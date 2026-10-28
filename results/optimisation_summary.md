Model optimisation (Oct 2026)

What was wrong on new images
- The deep cloud detector (OmniCloudMask) missed many clouds it had not seen before: on clouds made
  by our simulator it found only 1-40% of the cloud, so those areas were never rebuilt. The calibrated
  rule detector found about 94%. On real images it is the other way round (deep has the higher recall).
- The reference image was colour-matched to the cloudy image with one straight line per band for the
  whole image. When light or season changes differently across the scene, the reference is too bright
  in some places and too dark in others.
- On real Sentinel-2 clouds MeghGAN was barely better than plain copying (32.25 vs 32.24 dB).
- Two different default models (one per sensor).

What was changed
1. Hybrid cloud mask: deep mask OR calibrated rule mask (meghmukt/masking.py, detect_auto "hybrid").
2. Local colour matching of the reference: global fit, then a smooth correction from the clear pixels
   around each cloud at two scales (96 and 24 px) (meghmukt/dataset.py, match_reference).
3. Training that copes with real errors: random grow/shrink of the cloud mask by 1-4 px, and a
   reference shifted by up to 1.5 px with uneven brightness change (jitter_mask, perturb_reference).
4. One model for both sensors: trained 5000 steps on simulated + real clouds from LISS-IV and
   Sentinel-2 together (scripts/03_train.py --unified, 46 min), then averaged with the LISS-IV model
   weights (checkpoints/meghgan_opt.pt). Now the default for both sensors.

Held-out benchmark, PSNR on the cloudy area (dB). Areas never used in training.
13 real LISS-IV clouds, 20 simulated LISS-IV, 13 real Sentinel-2 clouds (6 unseen areas), 20 simulated Sentinel-2.

| | LISS-IV real | LISS-IV simulated | S2 real | S2 simulated |
|---|---|---|---|---|
| Temporal Fill, old matching | 30.58 | 30.19 | 32.24 | 34.83 |
| MeghGAN before (old default for the sensor, old matching) | 33.54 | 33.34 | 32.25 | 36.28 |
| MeghGAN after (meghgan_opt, local matching) | **34.22** | **33.59** | **32.57** | **36.43** |
| Gain | +0.68 | +0.25 | +0.32 | +0.15 |

Other models with local matching (for the choice of meghgan_opt):

| | LISS-IV real | LISS-IV simulated | S2 real | S2 simulated | mean |
|---|---|---|---|---|---|
| meghgan_real | 32.59 | 32.05 | 32.59 | 36.02 | 33.81 |
| meghgan_liss4 | 34.36 | 33.41 | 32.57 | 36.05 | 34.10 |
| meghgan_unified | 33.99 | 33.60 | 32.39 | 36.56 | 34.14 |
| meghgan_opt | 34.22 | 33.59 | 32.57 | 36.43 | 34.20 |

Without a reference image meghgan_opt is the best on all four sets
(35.08 / 32.26 / 29.01 / 32.14 dB vs 34.89 / 32.03 / 28.71 / 31.67 for meghgan_liss4).

Three test images with known truth (made from training images, run through the upload page of the app)

| Test | Cloudy input | Temporal Fill | MeghGAN (meghgan_opt) |
|---|---|---|---|
| 1 LISS-IV floodplain | 5.2 dB | 32.3 dB / SSIM 0.823 | 33.5 dB / SSIM 0.867 / SAM 4.30 |
| 2 LISS-IV villages | 19.0 dB | 33.0 dB / SSIM 0.745 | 33.9 dB / SSIM 0.812 / SAM 3.86 |
| 3 Sentinel-2 Kaziranga | 8.7 dB | 34.8 dB / SSIM 0.925 | 34.3 dB / SSIM 0.932 / SAM 3.07 |

Cloud detection on the test images (IoU with the true mask): deep only 0.01-0.40, hybrid 0.91-0.95.

Limits
- Very faint cloud shadows (< ~30% darker) can still be missed by the detector.
- When the reference is almost identical to the truth (Test 3), plain copying can score a slightly
  higher PSNR; MeghGAN keeps the best SSIM.
