Real clouds, 13 near-date pairs (gap <= 10 days, ground correlation >= 0.8), unseen areas.
NIQE of the near-date truth on the same pixels: {'deep': 1.3241006010588767, 'rules': 3.5192899275269016}

### mask: deep

| Model | PSNR cloud | SSIM cloud | MAE | SAM ° | NDVI MAE | Land-cover | Sharpness | NIQE ↓ |
|---|---|---|---|---|---|---|---|---|
| Do nothing | 15.77 | 0.472 | 0.1286 | 10.17 | 0.194 | 33.1% | 1.54 | 3.47 |
| M0 Classical temporal fill | 32.24 | 0.786 | 0.0181 | 5.11 | 0.101 | 52.6% | 0.83 | 1.67 |
| M1 U-Net baseline | 23.20 | 0.530 | 0.0517 | 12.64 | 0.252 | 32.6% | 1.10 | 3.07 |
| M2 MeghGAN (ours) | 26.82 | 0.733 | 0.0337 | 9.01 | 0.184 | 39.0% | 0.96 | 2.10 |
| M2 without GAN loss | 27.28 | 0.740 | 0.0314 | 8.83 | 0.182 | 39.3% | 0.96 | 2.00 |
| M2 + Charbonnier | 26.61 | 0.726 | 0.0352 | 9.05 | 0.187 | 38.9% | 0.95 | 2.06 |
| M2 MeghGAN + real-cloud training | 32.25 | 0.824 | 0.0185 | 4.47 | 0.087 | 55.5% | 0.88 | 1.67 |
| M3 Conditional diffusion | 27.46 | 0.726 | 0.0308 | 8.20 | 0.158 | 39.7% | 0.96 | 2.31 |
| M3 Diffusion + real-cloud training | 31.41 | 0.782 | 0.0196 | 5.21 | 0.101 | 49.1% | 0.87 | 2.23 |

### mask: rules

| Model | PSNR cloud | SSIM cloud | MAE | SAM ° | NDVI MAE | Land-cover | Sharpness | NIQE ↓ |
|---|---|---|---|---|---|---|---|---|
| Do nothing | 28.23 | 0.875 | 0.0291 | 3.66 | 0.072 | 79.0% | 1.15 | 3.50 |
| M0 Classical temporal fill | 28.34 | 0.789 | 0.0243 | 6.75 | 0.143 | 53.2% | 0.94 | 3.42 |
| M2 MeghGAN (ours) | 29.75 | 0.823 | 0.0260 | 4.64 | 0.100 | 62.8% | 1.34 | 3.84 |
| M2 MeghGAN + real-cloud training | 30.27 | 0.858 | 0.0237 | 3.93 | 0.083 | 73.4% | 1.11 | 3.44 |
