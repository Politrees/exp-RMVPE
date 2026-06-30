import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from src.utils import to_local_average_cents
from src.loss import bce
from src.constants import SAMPLE_RATE
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy, voicing_recall, voicing_false_alarm


def _freq_from_cents(cents: np.ndarray, voiced: np.ndarray | None = None) -> np.ndarray:
    cents = np.asarray(cents, dtype=np.float64)
    freq = np.zeros_like(cents, dtype=np.float64)

    if voiced is None:
        voiced = cents > 0

    voiced = np.asarray(voiced).astype(bool)
    valid = voiced & np.isfinite(cents) & (cents > 0)
    freq[valid] = 10.0 * (2.0 ** (cents[valid] / 1200.0))
    return freq


def _evaluate_voicing_detection(pred_voiced: np.ndarray, true_voiced: np.ndarray) -> dict:
    pred_voiced = pred_voiced.astype(bool)
    true_voiced = true_voiced.astype(bool)

    tp = np.sum(pred_voiced & true_voiced)
    fp = np.sum(pred_voiced & ~true_voiced)
    fn = np.sum(~pred_voiced & true_voiced)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _evaluate_pitch_accuracy(
    pred_freq: np.ndarray,
    true_freq: np.ndarray,
    pred_voiced: np.ndarray,
    true_voiced: np.ndarray,
    epsilon_cents: float = 50.0,
    gross_error_threshold: float = 200.0,
) -> dict:
    valid_mask = pred_voiced & true_voiced

    if not np.any(valid_mask):
        return {
            "rmse": np.nan,
            "cents_error": np.nan,
            "rpa": 0.0,
            "rca": 0.0,
            "octave_error_rate": 1.0,
            "gross_error_rate": 1.0,
            "valid_frames": 0,
        }

    pred = pred_freq[valid_mask]
    true = true_freq[valid_mask]

    eps = np.finfo(np.float64).eps
    with np.errstate(divide="ignore", invalid="ignore"):
        cents_diff = np.abs(1200.0 * np.log2((pred + eps) / (true + eps)))

    rpa = np.nanmean(cents_diff < epsilon_cents)

    wrapped_cents_diff = cents_diff % 1200.0
    chroma_diff = np.minimum(wrapped_cents_diff, 1200.0 - wrapped_cents_diff)
    rca = np.nanmean(chroma_diff < epsilon_cents)

    gross_error_rate = np.nanmean(cents_diff > gross_error_threshold)

    relative_error = np.abs(pred - true) / (true + eps)
    octave_errors = np.logical_or(relative_error > 0.4, (cents_diff > 1100.0) & (cents_diff < 1300.0),)
    octave_error_rate = np.nanmean(octave_errors)

    rmse = np.sqrt(np.nanmean((pred - true) ** 2))
    cents_error = np.nanmean(cents_diff)

    return {
        "rmse": float(rmse),
        "cents_error": float(cents_error),
        "rpa": float(rpa),
        "rca": float(rca),
        "octave_error_rate": float(octave_error_rate),
        "gross_error_rate": float(gross_error_rate),
        "valid_frames": int(np.sum(valid_mask)),
    }


def _calculate_combined_score(voicing_metrics: dict, pitch_metrics: dict) -> float:
    cents_error = pitch_metrics["cents_error"]
    octave_error_rate = pitch_metrics["octave_error_rate"]
    gross_error_rate = pitch_metrics["gross_error_rate"]

    cents_accuracy = 0.0 if not np.isfinite(cents_error) else float(np.exp(-cents_error / 500.0))
    octave_accuracy = float(np.exp(-10.0 * octave_error_rate))
    gross_error_accuracy = float(np.exp(-5.0 * gross_error_rate))

    components = [
        pitch_metrics["rpa"],
        cents_accuracy,
        voicing_metrics["precision"],
        voicing_metrics["recall"],
        octave_accuracy,
        gross_error_accuracy,
    ]

    valid_components = [c for c in components if np.isfinite(c) and c > 0]
    if not valid_components:
        return 0.0

    return float(len(valid_components) / sum(1.0 / c for c in valid_components))


@torch.inference_mode()
def evaluate(dataset, model, hop_length, device, pitch_th=0.03):
    metrics = defaultdict(list)

    for data in dataset:
        mel = data["mel"].to(device)
        n_frames = mel.shape[-1]

        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect").unsqueeze(0)

        output_chunks = []
        pad_frames = mel.shape[-1]

        for start in range(0, pad_frames, 32000):
            end = min(start + 32000, pad_frames)
            mel_chunk = mel[..., start:end]
            assert mel_chunk.shape[-1] % 32 == 0, "chunk_size must be divisible by 32"
            out_chunk = model(mel_chunk)
            output_chunks.append(out_chunk)

        pitch_pred = torch.cat(output_chunks, dim=1).squeeze(0)
        pitch_label = data["pitch"].to(device)

        voice_label = data.get("voice", None)
        cent_label = data.get("cent", None)

        min_len = min(pitch_pred.shape[0], pitch_label.shape[0])
        if voice_label is not None:
            min_len = min(min_len, voice_label.shape[0])
        if cent_label is not None:
            min_len = min(min_len, cent_label.shape[0])

        pitch_pred = pitch_pred[:min_len]
        pitch_label = pitch_label[:min_len]
        if voice_label is not None:
            voice_label = voice_label[:min_len]
        if cent_label is not None:
            cent_label = cent_label[:min_len]

        loss = bce(pitch_pred, pitch_label)
        metrics["loss"].append(loss.item())

        pitch_pred_np = pitch_pred.detach().cpu().numpy()
        cents_pred = to_local_average_cents(pitch_pred_np, None, pitch_th)
        pred_voiced = np.max(pitch_pred_np, axis=1) > pitch_th

        if cent_label is not None:
            cents_label = cent_label.detach().cpu().numpy()
        else:
            cents_label = to_local_average_cents(pitch_label.detach().cpu().numpy(), None, pitch_th)

        if voice_label is not None:
            true_voiced = voice_label.detach().cpu().numpy().astype(bool)
        else:
            true_voiced = cents_label > 0

        freq_pred = _freq_from_cents(cents_pred, pred_voiced)
        freq_true = _freq_from_cents(cents_label, true_voiced)

        time_slice = np.arange(len(cents_label), dtype=np.float64) * hop_length / SAMPLE_RATE
        ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq_true, time_slice, freq_pred)

        rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
        rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
        oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
        vfa = voicing_false_alarm(ref_v, est_v)
        vr = voicing_recall(ref_v, est_v)

        metrics["RPA"].append(float(rpa))
        metrics["RCA"].append(float(rca))
        metrics["OA"].append(float(oa))
        metrics["VFA"].append(float(vfa))
        metrics["VR"].append(float(vr))

        voicing_metrics = _evaluate_voicing_detection(pred_voiced, true_voiced)
        pitch_metrics = _evaluate_pitch_accuracy(freq_pred, freq_true, pred_voiced, true_voiced)
        hm = _calculate_combined_score(voicing_metrics, pitch_metrics)

        cents_error = pitch_metrics["cents_error"]
        ca = 0.0 if not np.isfinite(cents_error) else float(np.exp(-cents_error / 500.0))

        metrics["Precision"].append(voicing_metrics["precision"])
        metrics["Recall"].append(voicing_metrics["recall"])
        metrics["F1"].append(voicing_metrics["f1"])
        metrics["CA"].append(ca)
        metrics["RMSE_Hz"].append(pitch_metrics["rmse"])
        metrics["CentsError"].append(pitch_metrics["cents_error"])
        metrics["OctaveError"].append(pitch_metrics["octave_error_rate"])
        metrics["GrossError"].append(pitch_metrics["gross_error_rate"])
        metrics["HM"].append(hm)

        print(data["file"], ":\tRPA=", f"{rpa:.4f}", "\tOA=", f"{oa:.4f}", "\tHM=", f"{hm:.4f}")

    return metrics
