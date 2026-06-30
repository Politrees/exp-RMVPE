import os
import re
import sys
import argparse
import traceback
import datetime
from time import time as ttime

import torch
import numpy as np
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import E2E0
from src.utils import summary, cycle
from src.loss import bce
from src.dataset import HybridPitchDataset
from evaluate import evaluate

now_dir = os.getcwd()
sys.path.append(now_dir)


class IterRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = round(now_time - self.last_time, 1)
        self.last_time = now_time
        return f"[{str(datetime.timedelta(seconds=int(elapsed_time)))}]"


def find_latest_iteration(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pt")]
    iterations = []
    for f in model_files:
        match = re.search(r"model_(\d+)\.pt", f)
        if match:
            iterations.append(int(match.group(1)))

    return max(iterations) if iterations else None


def summarize_eval_metrics(metrics: dict) -> dict:
    summary = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        summary[key] = float(arr.mean()) if len(arr) > 0 else 0.0
    return summary


def ensure_result_header(result_path: str):
    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        return

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(
            "iter\tHM\tRPA\tRCA\tOA\tVR\tVFA\t"
            "Precision\tRecall\tF1\tCA\tCentsError\tRMSE_Hz\tOctaveError\tGrossError\n"
        )


def train(
    model_name,
    batch_size,
    dataset_dir,
    exp_root,
    iterations=100000,
    validation_interval=1000,
    num_workers=2,
):
    print("Начало обучения модели:", model_name, flush=True)

    experiment_dir = os.path.join(exp_root, model_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    tb_dir = os.path.join(experiment_dir, "tb")
    result_path = os.path.join(experiment_dir, "result.txt")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    ensure_result_header(result_path)

    hop_length = 160
    optimizer_type = "adamw"
    learning_rate = 5e-4
    log_interval = 50
    clip_grad_norm_value = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only_latest = False

    train_dataset = HybridPitchDataset(dataset_dir, hop_length, ["train"], whole_audio=False, use_aug=True)
    validation_dataset = HybridPitchDataset(dataset_dir, hop_length, ["test"], whole_audio=True, use_aug=False)

    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
    )

    resume_path = None
    if only_latest:
        potential_path = os.path.join(checkpoint_dir, "model_latest.pt")
        if os.path.exists(potential_path):
            resume_path = potential_path
    else:
        latest_iter = find_latest_iteration(checkpoint_dir)
        if latest_iter is not None:
            resume_path = os.path.join(checkpoint_dir, f"model_{latest_iter}.pt")
        else:
            latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
            if os.path.exists(latest_path):
                resume_path = latest_path

    should_resume = resume_path is not None and os.path.exists(resume_path)
    resume_iteration = 0

    writer = SummaryWriter(tb_dir)

    model = E2E0(1, 1, 16).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-6)

    best_hm = 0.0
    best_rpa = 0.0

    if should_resume:
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        state_dict = ckpt["model"]

        if isinstance(model, nn.DataParallel):
            if list(state_dict.keys())[0].startswith("module."):
                model.load_state_dict(state_dict)
            else:
                new_state_dict = {"module." + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
        else:
            if list(state_dict.keys())[0].startswith("module."):
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"Не удалось загрузить optimizer state: {e}", flush=True)

        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"Не удалось загрузить scheduler state: {e}", flush=True)

        resume_iteration = ckpt.get("iteration", 0)
        best_hm = ckpt.get("best_hm", ckpt.get("best_rpa", 0.0))
        best_rpa = ckpt.get("best_rpa", 0.0)

    if not isinstance(model, nn.DataParallel):
        summary(model)

    print(f"Обучение: {resume_iteration} → {iterations} итераций", flush=True)
    print(f"Начальный LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    iterrec = IterRecorder()
    model.train()

    for i, data in zip(range(resume_iteration + 1, iterations + 1), cycle(data_loader)):
        mel = data["mel"].to(device)
        pitch_label = data["pitch"].to(device)

        pitch_pred = model(mel)

        if pitch_pred.shape != pitch_label.shape:
            raise RuntimeError(
                "Shape mismatch between prediction and label: "
                f"pitch_pred.shape={tuple(pitch_pred.shape)}, "
                f"pitch_label.shape={tuple(pitch_label.shape)}, "
                f"mel.shape={tuple(mel.shape)}"
            )

        loss = bce(pitch_pred, pitch_label)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected: "
                f"loss={loss.item()}, "
                f"pitch_pred_min={pitch_pred.detach().min().item()}, "
                f"pitch_pred_max={pitch_pred.detach().max().item()}, "
                f"pitch_label_min={pitch_label.detach().min().item()}, "
                f"pitch_label_max={pitch_label.detach().max().item()}"
            )

        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm_value:
            clip_grad_norm_(model.parameters(), clip_grad_norm_value)

        optimizer.step()
        scheduler.step()

        if i % log_interval == 0:
            writer.add_scalar("loss/loss_pitch", loss.item(), global_step=i)
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", lr, global_step=i)
            writer.flush()

            print(f"{iterrec.record()}: Iter {i}/{iterations} | Loss: {loss.item():.6f} | LR: {lr:.2e}", flush=True)

        if i % validation_interval == 0:
            model.eval()

            with torch.no_grad():
                eval_model = model.module if isinstance(model, nn.DataParallel) else model
                metrics = evaluate(validation_dataset, eval_model, hop_length, device)
                summary_metrics = summarize_eval_metrics(metrics)

                for key, value in summary_metrics.items():
                    writer.add_scalar(f"stage_pitch/{key}", value, global_step=i)

                writer.flush()

                hm = summary_metrics.get("HM", 0.0)
                rpa = summary_metrics.get("RPA", 0.0)
                rca = summary_metrics.get("RCA", 0.0)
                oa = summary_metrics.get("OA", 0.0)
                vr = summary_metrics.get("VR", 0.0)
                vfa = summary_metrics.get("VFA", 0.0)
                precision = summary_metrics.get("Precision", 0.0)
                recall = summary_metrics.get("Recall", 0.0)
                f1 = summary_metrics.get("F1", 0.0)
                ca = summary_metrics.get("CA", 0.0)
                cents_error = summary_metrics.get("CentsError", 0.0)
                rmse_hz = summary_metrics.get("RMSE_Hz", 0.0)
                octave_error = summary_metrics.get("OctaveError", 0.0)
                gross_error = summary_metrics.get("GrossError", 0.0)

                print(
                    f"=== Validation @ {i} | "
                    f"HM: {hm:.4f} | RPA: {rpa:.4f} | F1: {f1:.4f} | "
                    f"CA: {ca:.4f} | OA: {oa:.4f} ===",
                    flush=True,
                )

                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{i}\t"
                        f"{hm}\t{rpa}\t{rca}\t{oa}\t{vr}\t{vfa}\t"
                        f"{precision}\t{recall}\t{f1}\t{ca}\t"
                        f"{cents_error}\t{rmse_hz}\t{octave_error}\t{gross_error}\n"
                    )

                is_best = False
                if hm >= best_hm:
                    best_hm = hm
                    best_rpa = max(best_rpa, rpa)
                    is_best = True
                    print(f"New best model at {i}! (HM={hm:.4f})", flush=True)

                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                checkpoint_dict = {
                    "iteration": i,
                    "model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_hm": best_hm,
                    "best_rpa": best_rpa,
                }

                if is_best:
                    torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "model_best.pt"))

                if only_latest:
                    torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "model_latest.pt"))
                else:
                    torch.save(checkpoint_dict, os.path.join(checkpoint_dir, f"model_{i}.pt"))

            model.train()

    print("Training finished.", flush=True)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True, help="Путь к dataset с папками train/test")
    parser.add_argument("--exp_root", type=str, required=True, help="Корень для логов и чекпоинтов")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--validation_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    try:
        train(
            args.name,
            args.batch_size,
            args.dataset_dir,
            args.exp_root,
            iterations=args.iterations,
            validation_interval=args.validation_interval,
            num_workers=args.num_workers,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.", flush=True)
        raise
    except Exception:
        print("Training failed with exception:", flush=True)
        traceback.print_exc()
        raise
