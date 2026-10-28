import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt import losses as L
from meghmukt.dataset import (FixedSet, MixedDataset, MultiMix, Normaliser, RealPairDataset,
                              SyntheticCloudDataset)
from meghmukt.io import get_device, load_config
from meghmukt.metrics import psnr
from meghmukt.models import PatchDiscriminator, build, save_checkpoint, soft_mask
from meghmukt.synth import CloudSynthesiser


def make_sources(cfg, finetune):
    P = cfg["paths"]["processed"]
    if finetune:
        d = np.load(P / "liss4.npz")
        return [{"clear": x[None].astype(np.float16)} for x in d["train"]]
    return [{"clear": np.load(P / f"s2_{a}.npz")["clear"]} for a in cfg["split"]["train"]]


@torch.no_grad()
def validate(model, name, loader, norm, device, max_batches=6):
    max_batches = len(loader) if isinstance(loader.dataset, ConcatDataset) else max_batches
    model.eval()
    scores = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        cond, gt = batch[0].to(device), batch[1]
        pred = model.sample(cond, steps=20) if name == "diffusion" else model(cond)
        pred, gt = norm.inverse(pred).cpu().numpy(), norm.inverse(gt).numpy()
        m = (soft_mask(cond, blur=0) > 0.5).cpu().numpy()[:, 0]
        if len(batch) > 2:
            m = m & (batch[2].numpy() > 0.5)
        scores += [psnr(p, g, mm) for p, g, mm in zip(pred, gt, m) if mm.sum() > 20]
    model.train()
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "meghgan", "diffusion"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--no-spectral", action="store_true", help="ablation: drop SAM + NDVI losses")
    ap.add_argument("--no-adv", action="store_true", help="ablation: no discriminator / adversarial loss")
    ap.add_argument("--charbonnier", action="store_true", help="Charbonnier instead of L1 pixel loss")
    ap.add_argument("--finetune", help="checkpoint to fine-tune on real LISS-IV")
    ap.add_argument("--real-finetune", help="checkpoint to fine-tune on real-cloud Sentinel-2 pairs (+ synthetic)")
    ap.add_argument("--liss4", help="checkpoint to fine-tune on real Bhoonidhi LISS-IV (+ Sentinel-2 to keep)")
    ap.add_argument("--unified", help="checkpoint to train into one model for both sensors, with mask/reference noise")
    args = ap.parse_args()

    cfg = load_config()
    tc = cfg["train"]
    P = cfg["paths"]
    torch.manual_seed(tc["seed"]), np.random.seed(tc["seed"])
    device = get_device(tc["device"])
    tag = args.tag or (args.model + ("_liss4ft" if args.finetune else "_real" if args.real_finetune
                                     else "_liss4" if args.liss4 else "_unified" if args.unified else ""))
    steps = args.steps or (tc["finetune"]["steps"] if args.finetune else 5000 if args.unified
                           else 2500 if (args.real_finetune or args.liss4) else tc[args.model]["steps"])
    lr = tc["finetune"]["lr"] if args.finetune else 1e-4 if (args.real_finetune or args.liss4 or args.unified) else tc["lr"]
    init = args.finetune or args.real_finetune or args.liss4 or args.unified

    norm = Normaliser.load(P["processed"] / "norm.json")
    synth = CloudSynthesiser(np.load(P["processed"] / "cloud_spectra.npy"), tuple(cfg["data"]["cloud_cover_range"]),
                             cfg["data"]["shadow_prob"])
    ref_dropout = 0.5 if args.finetune else cfg["data"]["reference_dropout"]
    train_ds = SyntheticCloudDataset(make_sources(cfg, args.finetune), norm, synth, cfg["data"]["patch_size"],
                                     length=steps * tc["batch_size"], ref_dropout=ref_dropout, seed=tc["seed"])
    if args.real_finetune:
        real = RealPairDataset(P["processed"] / "realpairs_train.npz", norm, cfg["data"]["patch_size"],
                               ref_dropout=cfg["data"]["reference_dropout"], seed=tc["seed"])
        train_ds = MixedDataset(real, train_ds, p_real=0.5, length=steps * tc["batch_size"], seed=tc["seed"])
        val_ds = FixedSet(P["processed"] / "eval_real_val.npz", with_valid=True)
    elif args.liss4:
        lc = np.load(P["processed"] / "liss4_clear_train.npz")
        liss_syn = SyntheticCloudDataset([{"clear": lc[k]} for k in lc.files], norm, synth, cfg["data"]["patch_size"],
                                         ref_dropout=cfg["data"]["reference_dropout"], seed=tc["seed"] + 1)
        liss_real = RealPairDataset(P["processed"] / "realpairs_liss4_train.npz", norm, cfg["data"]["patch_size"],
                                    ref_dropout=cfg["data"]["reference_dropout"], seed=tc["seed"] + 2)
        s2_real = RealPairDataset(P["processed"] / "realpairs_train.npz", norm, cfg["data"]["patch_size"],
                                  ref_dropout=cfg["data"]["reference_dropout"], seed=tc["seed"] + 3)
        train_ds = MultiMix([(liss_syn, 0.35), (liss_real, 0.30), (train_ds, 0.20), (s2_real, 0.15)],
                            length=steps * tc["batch_size"], seed=tc["seed"])
        val_ds = FixedSet(P["processed"] / "eval_liss4_val.npz")
    elif args.unified:
        rd, seed = cfg["data"]["reference_dropout"], tc["seed"]
        lc = np.load(P["processed"] / "liss4_clear_train.npz")
        parts = [
            (SyntheticCloudDataset([{"clear": lc[k]} for k in lc.files], norm, synth, cfg["data"]["patch_size"],
                                   ref_dropout=rd, seed=seed + 1, robust=True), 0.30),
            (RealPairDataset(P["processed"] / "realpairs_liss4_train.npz", norm, cfg["data"]["patch_size"],
                             ref_dropout=rd, seed=seed + 2, robust=True), 0.25),
            (SyntheticCloudDataset(make_sources(cfg, False), norm, synth, cfg["data"]["patch_size"],
                                   ref_dropout=rd, seed=seed + 3, robust=True), 0.25),
            (RealPairDataset(P["processed"] / "realpairs_train.npz", norm, cfg["data"]["patch_size"],
                             ref_dropout=rd, seed=seed + 4, robust=True), 0.20)]
        train_ds = MultiMix(parts, length=steps * tc["batch_size"], seed=seed)
        val_ds = ConcatDataset([FixedSet(P["processed"] / "eval_liss4_val.npz", with_valid=True, rematch=norm),
                                FixedSet(P["processed"] / "eval_real_val.npz", with_valid=True, rematch=norm)])
    else:
        val_ds = FixedSet(P["processed"] / "eval_val.npz")
    loader = DataLoader(train_ds, tc["batch_size"], num_workers=3 if (args.liss4 or args.unified) else 4, persistent_workers=True,
                        drop_last=True)
    val_loader = DataLoader(val_ds, 16, shuffle=False)

    model = build(args.model, cfg).to(device)
    if init:
        model.load_state_dict(torch.load(init, map_location=device, weights_only=False)["state"])
    opt = torch.optim.AdamW(model.parameters(), lr, betas=(0.5 if args.model == "meghgan" else 0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    disc = None
    if args.model == "meghgan" and not args.no_adv:
        disc = PatchDiscriminator().to(device)
        opt_d = torch.optim.AdamW(disc.parameters(), lr, betas=(0.5, 0.999))
    if args.model == "meghgan":
        g = tc["meghgan"]
        lam = dict(l1=g["lambda_l1"], ssim=g["lambda_ssim"], sam=0 if args.no_spectral else g["lambda_sam"],
                   ndvi=0 if args.no_spectral else g["lambda_ndvi"], adv=0 if args.no_adv else g["lambda_adv"])
    else:
        lam = dict(l1=10.0, ssim=2.0)

    log_dir = P["results"] / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{tag}.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["step", "loss", "loss_d", "val_psnr_cloud", "seconds"])
    ckpt = P["checkpoints"] / f"{tag}.pt"
    best, t0, run, run_d = -1.0, time.time(), [], []
    val_every = max(steps // 15, 50)
    print(f"Training {tag} on {device} for {steps} steps (val every {val_every}) ...", flush=True)

    model.train()
    for step, batch in enumerate(loader, 1):
        cond, gt = batch[0].to(device), batch[1].to(device)
        valid = batch[2].to(device)[:, None] if len(batch) > 2 else None
        if args.model == "diffusion":
            loss = model.loss(gt, cond, valid)
        else:
            pred = model(cond)
            pp, gg = norm.inverse(pred), norm.inverse(gt)
            m = soft_mask(cond) * (1 if valid is None else valid)
            loss = (lam["l1"] * L.pixel_loss(pp, gg, m, args.charbonnier, valid=valid)
                    + lam["ssim"] * L.ssim_loss(pp, gg, valid))
            if lam.get("sam"):
                loss = loss + lam["sam"] * L.sam_loss(pp, gg, m) + lam["ndvi"] * L.ndvi_loss(pp, gg, m)
            if disc is not None:
                d_loss = 0.5 * (L.gan_loss(disc(gt, cond), True) + L.gan_loss(disc(pred.detach(), cond), False))
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                run_d.append(d_loss.item())
                loss = loss + lam["adv"] * L.gan_loss(disc(pred, cond), True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())

        if step % val_every == 0 or step == steps:
            v = validate(model, args.model, val_loader, norm, device)
            wr.writerow([step, np.mean(run), np.mean(run_d) if run_d else "", v, round(time.time() - t0)])
            log.flush()
            flag = ""
            if v > best or args.finetune:
                best, flag = v, "  *saved*"
                save_checkpoint(ckpt, model, args.model, norm.to_dict(),
                                {"tag": tag, "step": step, "val_psnr_cloud": v, "lambdas": lam,
                                 "finetuned_from": init, "real_clouds": bool(args.real_finetune or args.liss4 or args.unified),
                                 "liss4": bool(args.liss4), "unified": bool(args.unified),
                                 "charbonnier": args.charbonnier})
            print(f"step {step:6d}  loss {np.mean(run):.4f}  val PSNR(cloud) {v:.2f} dB  "
                  f"{(time.time() - t0) / 60:.1f} min{flag}", flush=True)
            run, run_d = [], []
        if step >= steps:
            break
    log.close()
    (P["results"] / "train_logs" / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "model": args.model, "steps": steps, "best_val_psnr_cloud": best,
         "minutes": round((time.time() - t0) / 60, 1), "device": str(device), "lambdas": lam}, indent=2))
    print(f"Done. best val PSNR(cloud) = {best:.2f} dB -> {ckpt}")


if __name__ == "__main__":
    main()
