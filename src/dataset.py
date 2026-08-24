import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import colorednoise as cn
except Exception:  # pragma: no cover
    cn = None

from .constants import CONST, MEL_FMAX, MEL_FMIN, N_CLASS, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from .spec import MelSpectrogram


class HybridPitchDataset(Dataset):
    AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aiff", ".aif")
    LABEL_EXTS = (".pv", ".csv", ".tsv", ".txt", ".lab")

    TIME_COLUMNS = ("time", "times", "timestamp", "t", "sec", "seconds")
    START_COLUMNS = ("onset", "start", "begin", "start_time", "start_sec", "start_s")
    END_COLUMNS = ("offset", "end", "end_time", "end_sec", "end_s", "stop")
    DURATION_COLUMNS = ("duration", "dur", "length")
    MIDI_COLUMNS = ("midi", "note", "pitch", "midi_note")
    HZ_COLUMNS = ("f0", "freq", "frequency", "hz", "pitch_hz", "fundamental")
    CENT_COLUMNS = ("cent", "cents")
    VOICE_COLUMNS = ("voice", "voiced", "vuv", "uv", "is_voiced")
    CONF_COLUMNS = ("confidence", "conf", "prob", "probability")

    def __init__(
        self,
        path: str,
        hop_length: int = 160,
        groups: Optional[Iterable[str]] = None,
        whole_audio: bool = False,
        use_aug: bool = True,
        segment_frames: int = 256,
        min_frames: Optional[int] = None,
        label_unit: str = "auto",  # auto | midi | hz | cent
        key_shift_range: Tuple[float, float] = (-12.0, 12.0),
        noise_beta_range: Tuple[float, float] = (0.0, 2.0),
        noise_amp_log10_range: Tuple[float, float] = (-6.0, -1.0),
        volume_log10_range: Tuple[float, float] = (-1.0, 1.0),
        gaussian_sigma: float = 1.25,
        f0_min: float = 30.0,
        f0_max: float = 2000.0,
        allow_missing_labels: bool = False,
        recursive: bool = False,
        verbose: bool = True,
        label_dir: Optional[str] = None,
    ):
        super().__init__()
        self.path = path
        self.label_dir = os.path.abspath(label_dir) if label_dir else None
        self.HOP_LENGTH = int(hop_length)
        self.hop_length = int(hop_length)
        self.num_class = N_CLASS
        self.whole_audio = bool(whole_audio)
        self.use_aug = bool(use_aug)
        self.segment_frames = int(segment_frames)
        self.min_frames = int(min_frames) if min_frames is not None else self.segment_frames
        self.label_unit = label_unit.lower()
        self.key_shift_range = key_shift_range
        self.noise_beta_range = noise_beta_range
        self.noise_amp_log10_range = noise_amp_log10_range
        self.volume_log10_range = volume_log10_range
        self.gaussian_sigma = float(gaussian_sigma)
        self.f0_min = float(f0_min)
        self.f0_max = float(f0_max)
        self.allow_missing_labels = bool(allow_missing_labels)
        self.recursive = bool(recursive)
        self.verbose = bool(verbose)

        self.paths: List[str] = []
        self.data_buffer: Dict[str, Dict[str, object]] = {}
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, self.HOP_LENGTH, None, MEL_FMIN, MEL_FMAX)

        if groups is None:
            groups = [""]
        groups = list(groups)

        if self.verbose:
            print(
                f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
                f"of {self.__class__.__name__} at {path}"
            )

        for group in groups:
            pairs = self.files(group)
            iterator = tqdm(pairs, desc=f"Loading group {group}") if self.verbose else pairs
            for audio_path, label_path in iterator:
                try:
                    self.load(audio_path, label_path)
                except Exception as exc:
                    if self.allow_missing_labels:
                        print(f"[WARN] skip {audio_path}: {exc}")
                    else:
                        raise

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        audio_path = self.paths[index]
        data = self.data_buffer[audio_path]

        n = int(data["len"])
        audio = data["audio"]
        noise = data["noise"]
        cent = data["cent"]
        voice = data["voice"]

        if not isinstance(audio, torch.Tensor):
            raise TypeError("Internal audio buffer must be torch.Tensor")

        # Гарантируем минимальную длину для random crop.
        if n < self.min_frames:
            pad_n = self.min_frames - n
            pad_s = pad_n * self.HOP_LENGTH

            natural_silence = self._generate_natural_silence(pad_s, noise_level="low")
            audio = torch.cat([audio[:-WINDOW_LENGTH], natural_silence, torch.zeros(WINDOW_LENGTH)])

            if isinstance(noise, torch.Tensor):
                noise_silence = self._generate_natural_silence(pad_s, noise_level="medium")
                noise = torch.cat([noise[:-WINDOW_LENGTH], noise_silence, torch.zeros(WINDOW_LENGTH)])
            else:
                noise = None

            cent = F.pad(cent, (0, pad_n), mode="constant")
            voice = F.pad(voice, (0, pad_n), mode="constant")
            n = self.min_frames

        if self.whole_audio:
            start_frame = 0
            end_frame = n
        else:
            max_start = max(0, n - self.segment_frames)
            start_frame = random.randint(0, max_start)
            end_frame = start_frame + self.segment_frames

        if self.use_aug:
            key_shift = random.uniform(*self.key_shift_range)
            # Верхний регистр (сопрано/свист) почти синусоидальный, и именно
            # там RMVPE ошибается на октаву. Чтобы модель увидела такой регистр
            # даже если датасет вокальный и ниже C6, в ~25% кадров переносим
            # сегмент на октаву/две вверх. Это тот же mel-keyshift, что и выше;
            # метки корректируются автоматически через win_length (ниже).
            if random.random() < 0.25:
                key_shift += random.choice((12.0, 24.0))
        else:
            key_shift = 0.0

        factor = 2 ** (key_shift / 12.0)
        win_length_new = max(16, int(np.round(WINDOW_LENGTH * factor)))

        start_id = WINDOW_LENGTH + start_frame * self.HOP_LENGTH - win_length_new // 2
        end_id = WINDOW_LENGTH + (end_frame - 1) * self.HOP_LENGTH + (win_length_new + 1) // 2

        # На всякий случай расширяем буфер, если агрессивный key_shift вышел за границы.
        if start_id < 0 or end_id > len(audio):
            left_pad = max(0, -start_id)
            right_pad = max(0, end_id - len(audio))
            audio = F.pad(audio, (left_pad, right_pad), mode="constant")
            if isinstance(noise, torch.Tensor):
                noise = F.pad(noise, (left_pad, right_pad), mode="constant")
            start_id += left_pad
            end_id += left_pad

        aud = audio[start_id:end_id]

        if self.use_aug:
            if isinstance(noise, torch.Tensor):
                noi = random.uniform(-1.0, 1.0) * noise[start_id:end_id]
            else:
                noi = self._generate_colored_noise(len(aud))

            audio_aug = aud + noi
            max_amp = float(torch.max(torch.abs(audio_aug))) + 1e-5
            max_shift = min(self.volume_log10_range[1], np.log10(1.0 / max_amp))
            min_shift = self.volume_log10_range[0]
            if max_shift < min_shift:
                log10_vol_shift = max_shift
            else:
                log10_vol_shift = random.uniform(min_shift, max_shift)
            audio_aug = audio_aug * (10 ** log10_vol_shift)
        else:
            audio_aug = aud + (noise[start_id:end_id] if isinstance(noise, torch.Tensor) else 0)

        audio_aug = torch.clamp(audio_aug, -1.0, 1.0)
        mel = self.mel(audio_aug.unsqueeze(0), keyshift=key_shift, center=False).squeeze(0)

        target_frames = end_frame - start_frame
        if mel.shape[-1] > target_frames:
            mel = mel[:, :target_frames]
        elif mel.shape[-1] < target_frames:
            mel = F.pad(mel, (0, target_frames - mel.shape[-1]), mode="constant", value=0.0)

        c = cent[start_frame:end_frame].clone()
        v = voice[start_frame:end_frame].clone()

        # Коррекция label под key-shift-аугментацию через изменение win_length.
        if key_shift != 0:
            c = c + 1200.0 * np.log2(win_length_new / WINDOW_LENGTH)

        index_float = (c - CONST) / 20.0
        valid = (v > 0) & (index_float >= 0) & (index_float < N_CLASS)
        v = v * valid.float()

        class_idx = torch.arange(N_CLASS, dtype=torch.float32).expand(end_frame - start_frame, -1)
        pitch_label = torch.exp(-((class_idx - index_float.unsqueeze(-1)) ** 2) / (2.0 * self.gaussian_sigma ** 2))
        pitch_label = pitch_label * v.unsqueeze(-1)

        return {
            "mel": mel,
            "pitch": pitch_label,
            "voice": v,
            "cent": c,
            "file": audio_path,
            "label": data.get("label"),
        }

    @staticmethod
    def available_groups():
        return ["train", "test", "valid", "validation"]

    def files(self, group: str):
        group_path = os.path.join(self.path, group) if group else self.path
        glob_pattern = "**/*" if self.recursive else "*"

        audio_files: List[str] = []
        for ext in self.AUDIO_EXTS:
            audio_files.extend(glob(os.path.join(group_path, glob_pattern + ext), recursive=self.recursive))
            audio_files.extend(glob(os.path.join(group_path, glob_pattern + ext.upper()), recursive=self.recursive))

        audio_files = sorted(set(audio_files))
        pairs: List[Tuple[str, Optional[str]]] = []
        missing: List[str] = []

        group_label_dir = self.label_dir
        if group_label_dir and group:
            group_label_dir = os.path.join(group_label_dir, group)

        for audio_path in audio_files:
            label_path = self._find_label_for_audio(audio_path, label_dir=group_label_dir)
            if label_path is None:
                missing.append(audio_path)
                if self.allow_missing_labels:
                    pairs.append((audio_path, None))
            else:
                pairs.append((audio_path, label_path))

        if missing and not self.allow_missing_labels:
            examples = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"No label files found for {len(missing)} audio file(s). Examples:\n{examples}"
            )

        if self.verbose:
            print(f"Found {len(pairs)} audio/label pair(s) in {group_path}")
            if missing and self.allow_missing_labels:
                print(f"[WARN] {len(missing)} audio file(s) without labels will use empty labels")

        return pairs

    def load(self, audio_path: str, label_path: Optional[str]):
        wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)

        # Если файл двухканальный, сохраняем совместимость со старым HYBRID:
        # channel 0 = noise/accompaniment, channel 1 = target vocal.
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            if wav.shape[0] >= 2:
                noise_np = wav[0]
                audio_np = wav[1]
            else:
                noise_np = None
                audio_np = wav[0]
        else:
            noise_np = None
            audio_np = wav

        audio_np = np.asarray(audio_np, dtype=np.float32)
        audio_len = len(audio_np)
        n_frames = audio_len // self.HOP_LENGTH + 1

        audio_np = self._safe_reflect_pad(audio_np, WINDOW_LENGTH, WINDOW_LENGTH)
        audio = torch.from_numpy(audio_np).float()

        if noise_np is not None:
            noise_np = np.asarray(noise_np, dtype=np.float32)
            if len(noise_np) != audio_len:
                min_len = min(len(noise_np), audio_len)
                tmp = np.zeros(audio_len, dtype=np.float32)
                tmp[:min_len] = noise_np[:min_len]
                noise_np = tmp
            noise_np = self._safe_reflect_pad(noise_np, WINDOW_LENGTH, WINDOW_LENGTH)
            noise = torch.from_numpy(noise_np).float()
        else:
            noise = None

        if label_path is None:
            cent = torch.zeros(n_frames, dtype=torch.float32)
            voice = torch.zeros(n_frames, dtype=torch.float32)
        else:
            cent_np, voice_np = self._load_label(label_path, audio_path, n_frames)
            cent = torch.from_numpy(cent_np.astype(np.float32))
            voice = torch.from_numpy(voice_np.astype(np.float32))

        self.paths.append(audio_path)
        self.data_buffer[audio_path] = {
            "len": n_frames,
            "audio": audio,
            "noise": noise,
            "cent": cent,
            "voice": voice,
            "label": label_path,
        }

    # ------------------------- label discovery -------------------------

    def _find_label_for_audio(self, audio_path: str, label_dir: Optional[str] = None) -> Optional[str]:
        p = Path(audio_path)
        stem = p.stem

        candidate_stems = []
        for s in (stem, stem.replace("_m", ""), stem.replace("_p", ""), stem.replace("_vocal", "")):
            if s and s not in candidate_stems:
                candidate_stems.append(s)

        # 0) отдельный каталог с метками (MIR-1K хранит Wavfile/ и PitchLabel/ раздельно)
        search_label_dir = self.label_dir if label_dir is None else label_dir
        if search_label_dir is not None:
            lp = Path(search_label_dir)
            for s in candidate_stems:
                for ext in self.LABEL_EXTS:
                    candidate = lp / f"{s}{ext}"
                    if candidate.exists():
                        return str(candidate)
                    candidate = lp / f"{s}{ext.upper()}"
                    if candidate.exists():
                        return str(candidate)
            local_labels = []
            for ext in self.LABEL_EXTS:
                local_labels.extend(lp.glob(f"{stem}*{ext}"))
                local_labels.extend(lp.glob(f"{stem}*{ext.upper()}"))
            if len(local_labels) == 1:
                return str(local_labels[0])

        # 1) рядом с аудио
        for s in candidate_stems:
            for ext in self.LABEL_EXTS:
                candidate = p.with_name(s + ext)
                if candidate.exists():
                    return str(candidate)
                candidate = p.with_name(s + ext.upper())
                if candidate.exists():
                    return str(candidate)

        # 2) частый случай: audio xxx_m.wav, label xxx.pv уже покрыт выше.
        # 3) если рядом есть единственный label с похожим stem prefix.
        local_labels = []
        for ext in self.LABEL_EXTS:
            local_labels.extend(p.parent.glob(f"{stem}*{ext}"))
            local_labels.extend(p.parent.glob(f"{stem}*{ext.upper()}"))
        if len(local_labels) == 1:
            return str(local_labels[0])

        return None

    # ------------------------- label loading -------------------------

    def _load_label(self, label_path: str, audio_path: str, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        ext = Path(label_path).suffix.lower()
        if ext in (".pv", ".txt", ".lab"):
            return self._load_text_like_label(label_path, audio_path, n_frames)
        if ext in (".csv", ".tsv"):
            return self._load_table_label(label_path, audio_path, n_frames)
        raise ValueError(f"Unsupported label extension: {label_path}")

    def _load_text_like_label(self, label_path: str, audio_path: str, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        rows: List[List[float]] = []
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace(",", " ").replace("\t", " ")
                parts = [x for x in line.split(" ") if x]
                vals = []
                for part in parts:
                    try:
                        vals.append(float(part))
                    except ValueError:
                        pass
                if vals:
                    rows.append(vals)

        if not rows:
            return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)

        # Один столбец: frame-wise pitch.
        if all(len(r) == 1 for r in rows):
            values = np.array([r[0] for r in rows], dtype=np.float64)
            unit = self._infer_unit(values, label_path, audio_path, column_name=None, default_for_pv="midi")
            f0 = self._values_to_hz(values, unit)
            f0 = self._fit_1d_to_frames(f0, n_frames)
            return self._f0_to_cent_voice(f0)

        # Два и более столбца: считаем first=time, second=value.
        times = np.array([r[0] for r in rows if len(r) >= 2], dtype=np.float64)
        values = np.array([r[1] for r in rows if len(r) >= 2], dtype=np.float64)
        unit = self._infer_unit(values, label_path, audio_path, column_name=None, default_for_pv="hz")
        f0_values = self._values_to_hz(values, unit)
        f0 = self._time_values_to_frames(times, f0_values, n_frames)
        return self._f0_to_cent_voice(f0)

    def _load_table_label(self, label_path: str, audio_path: str, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        if pd is None:
            raise ImportError("pandas is required for csv/tsv labels")

        sep = "\t" if Path(label_path).suffix.lower() == ".tsv" else None

        # Сначала пробуем header=0. Если колонки все Unnamed/числа — ниже будет fallback header=None.
        try:
            df = pd.read_csv(label_path, sep=sep, engine="python", comment="#")
        except Exception:
            df = pd.read_csv(label_path, sep=sep, engine="python", comment="#", header=None)

        if df.empty:
            return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)

        df = df.dropna(how="all")
        original_columns = list(df.columns)
        lower_cols = [str(c).strip().lower() for c in df.columns]
        df.columns = lower_cols

        # Если pandas принял первую строку данных за header и имена колонок похожи на числа — читаем без header.
        if all(self._is_float_string(str(c)) for c in original_columns):
            df = pd.read_csv(label_path, sep=sep, engine="python", comment="#", header=None)
            df = df.dropna(how="all")
            df.columns = [str(i) for i in range(len(df.columns))]

        # Приводим числовые колонки.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")

        if df.empty:
            return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)

        start_col = self._first_existing(df.columns, self.START_COLUMNS)
        end_col = self._first_existing(df.columns, self.END_COLUMNS)
        dur_col = self._first_existing(df.columns, self.DURATION_COLUMNS)
        time_col = self._first_existing(df.columns, self.TIME_COLUMNS)
        value_col, value_unit = self._find_pitch_column(df.columns)
        voice_col = self._first_existing(df.columns, self.VOICE_COLUMNS)
        conf_col = self._first_existing(df.columns, self.CONF_COLUMNS)

        # Headerless fallback.
        if value_col is None:
            numeric_cols = [c for c in df.columns if df[c].notna().any()]
            if len(numeric_cols) == 1:
                value_col = numeric_cols[0]
                value_unit = self._infer_unit(df[value_col].to_numpy(), label_path, audio_path, None, default_for_pv="midi")
            elif len(numeric_cols) == 2:
                time_col, value_col = numeric_cols[0], numeric_cols[1]
                value_unit = self._infer_unit(df[value_col].to_numpy(), label_path, audio_path, None, default_for_pv="hz")
            elif len(numeric_cols) >= 3:
                # Частый interval format без header: onset, offset, note/f0
                start_col, end_col, value_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
                value_unit = self._infer_unit(df[value_col].to_numpy(), label_path, audio_path, None, default_for_pv="midi")

        if value_col is None:
            raise ValueError(f"Could not find pitch column in {label_path}. Columns: {list(df.columns)}")

        if self.label_unit != "auto":
            value_unit = self.label_unit
        elif value_unit is None:
            value_unit = self._infer_unit(df[value_col].to_numpy(), label_path, audio_path, value_col, default_for_pv="hz")

        # Interval/note формат: onset+offset+value или onset+duration+value.
        if start_col is not None and (end_col is not None or dur_col is not None):
            f0 = np.zeros(n_frames, dtype=np.float64)
            starts = df[start_col].to_numpy(dtype=np.float64)
            if end_col is not None:
                ends = df[end_col].to_numpy(dtype=np.float64)
            else:
                ends = starts + df[dur_col].to_numpy(dtype=np.float64)
            values = df[value_col].to_numpy(dtype=np.float64)
            hz_values = self._values_to_hz(values, value_unit)

            for onset, offset, hz in zip(starts, ends, hz_values):
                if not np.isfinite(onset) or not np.isfinite(offset) or not np.isfinite(hz):
                    continue
                if hz <= 0:
                    continue
                left = int(round(onset * SAMPLE_RATE / self.HOP_LENGTH))
                right = int(round(offset * SAMPLE_RATE / self.HOP_LENGTH)) + 1
                left = max(0, min(n_frames, left))
                right = max(left, min(n_frames, right))
                f0[left:right] = hz
            return self._f0_to_cent_voice(f0)

        values = df[value_col].to_numpy(dtype=np.float64)
        hz_values = self._values_to_hz(values, value_unit)

        if voice_col is not None:
            voice_values = df[voice_col].to_numpy(dtype=np.float64)
            hz_values = np.where(voice_values > 0, hz_values, 0.0)
        if conf_col is not None:
            # Если confidence явно 0, считаем unvoiced. Не вводим threshold, чтобы не терять слабие валидные участки.
            conf_values = df[conf_col].to_numpy(dtype=np.float64)
            hz_values = np.where(conf_values > 0, hz_values, 0.0)

        # Frame-wise с time column.
        if time_col is not None:
            times = df[time_col].to_numpy(dtype=np.float64)
            f0 = self._time_values_to_frames(times, hz_values, n_frames)
        else:
            f0 = self._fit_1d_to_frames(hz_values, n_frames)

        return self._f0_to_cent_voice(f0)

    # ------------------------- conversion helpers -------------------------

    def _find_pitch_column(self, columns) -> Tuple[Optional[str], Optional[str]]:
        col = self._first_existing(columns, self.HZ_COLUMNS)
        if col is not None:
            return col, "hz"
        col = self._first_existing(columns, self.MIDI_COLUMNS)
        if col is not None:
            return col, "midi"
        col = self._first_existing(columns, self.CENT_COLUMNS)
        if col is not None:
            return col, "cent"
        return None, None

    @staticmethod
    def _first_existing(columns, names: Iterable[str]) -> Optional[str]:
        columns = list(columns)
        lower_to_original = {str(c).lower(): c for c in columns}
        for name in names:
            if name in lower_to_original:
                return lower_to_original[name]
        return None

    @staticmethod
    def _is_float_string(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    def _infer_unit(
        self,
        values: np.ndarray,
        label_path: str,
        audio_path: str,
        column_name: Optional[str] = None,
        default_for_pv: str = "midi",
    ) -> str:
        if self.label_unit != "auto":
            return self.label_unit

        name = " ".join([str(column_name or ""), Path(label_path).stem, Path(audio_path).stem]).lower()
        if any(x in name for x in ("cent", "cents")):
            return "cent"
        if any(x in name for x in ("f0", "freq", "frequency", "hz")):
            return "hz"
        if any(x in name for x in ("midi", "note", "pitch")):
            return "midi"

        positive = np.asarray(values, dtype=np.float64)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if len(positive) == 0:
            return default_for_pv

        med = float(np.median(positive))
        mx = float(np.max(positive))

        if med > 1000 or mx > 2000:
            return "cent"
        if mx <= 127:
            # Для .pv/.txt чаще это MIDI, для time/value csv чаще может быть Hz.
            return default_for_pv
        return "hz"

    def _values_to_hz(self, values: np.ndarray, unit: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = np.where(np.isfinite(values), values, 0.0)
        unit = unit.lower()

        if unit == "hz":
            f0 = values.copy()
        elif unit == "midi":
            f0 = np.zeros_like(values, dtype=np.float64)
            mask = values > 0
            f0[mask] = 440.0 * (2.0 ** ((values[mask] - 69.0) / 12.0))
        elif unit == "cent":
            f0 = np.zeros_like(values, dtype=np.float64)
            mask = values > 0
            f0[mask] = 10.0 * (2.0 ** (values[mask] / 1200.0))
        else:
            raise ValueError(f"Unknown label unit: {unit}")

        f0[(f0 < self.f0_min) | (f0 > self.f0_max)] = 0.0
        return f0

    def _f0_to_cent_voice(self, f0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f0 = np.asarray(f0, dtype=np.float64)
        f0 = np.where(np.isfinite(f0), f0, 0.0)
        f0[(f0 < self.f0_min) | (f0 > self.f0_max)] = 0.0

        voice = (f0 > 0).astype(np.float32)
        cent = np.zeros_like(f0, dtype=np.float32)
        mask = f0 > 0
        cent[mask] = (1200.0 * np.log2(f0[mask] / 10.0)).astype(np.float32)
        return cent.astype(np.float32), voice.astype(np.float32)

    def _fit_1d_to_frames(self, values: np.ndarray, n_frames: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = np.where(np.isfinite(values), values, 0.0)
        if len(values) == n_frames:
            return values
        if len(values) > n_frames:
            return values[:n_frames]
        return np.pad(values, (0, n_frames - len(values)), mode="constant")

    def _time_values_to_frames(self, times: np.ndarray, values_hz: np.ndarray, n_frames: int) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        values_hz = np.asarray(values_hz, dtype=np.float64)
        mask = np.isfinite(times) & np.isfinite(values_hz)
        times = times[mask]
        values_hz = values_hz[mask]

        if len(times) == 0:
            return np.zeros(n_frames, dtype=np.float64)

        order = np.argsort(times)
        times = times[order]
        values_hz = values_hz[order]

        # Удаляем duplicate times, оставляя последнее значение.
        unique_times, unique_indices = np.unique(times, return_index=True)
        if len(unique_times) != len(times):
            # np.unique возвращает первый индекс, для простоты усредним дубликаты.
            new_values = []
            for t in unique_times:
                new_values.append(np.mean(values_hz[times == t]))
            times = unique_times
            values_hz = np.array(new_values, dtype=np.float64)

        target_times = np.arange(n_frames, dtype=np.float64) * self.HOP_LENGTH / SAMPLE_RATE

        if len(times) == 1:
            out = np.zeros(n_frames, dtype=np.float64)
            idx = int(round(times[0] * SAMPLE_RATE / self.HOP_LENGTH))
            if 0 <= idx < n_frames:
                out[idx] = values_hz[0]
            return out

        interp_f0 = np.interp(target_times, times, values_hz, left=0.0, right=0.0)
        interp_voice = np.interp(target_times, times, (values_hz > 0).astype(np.float64), left=0.0, right=0.0)
        interp_f0[interp_voice < 0.5] = 0.0
        return interp_f0

    # ------------------------- augmentation helpers -------------------------

    def _generate_colored_noise(self, length: int) -> torch.Tensor:
        if length <= 0:
            return torch.zeros(0, dtype=torch.float32)
        if cn is None:
            noise = np.random.randn(length).astype(np.float32)
        else:
            beta = random.uniform(*self.noise_beta_range)
            noise = cn.powerlaw_psd_gaussian(beta, length).astype(np.float32)
        amp = 10 ** random.uniform(*self.noise_amp_log10_range)
        return torch.from_numpy(noise).float() * amp

    def _generate_natural_silence(self, length: int, noise_level: str = "low") -> torch.Tensor:
        if length <= 0:
            return torch.zeros(0, dtype=torch.float32)
        levels = {"low": -60.0, "medium": -50.0, "high": -40.0}
        db = levels.get(noise_level, -60.0)
        amp = 10 ** (db / 20.0)
        if cn is None:
            silence = np.random.randn(length).astype(np.float32)
        else:
            silence = cn.powerlaw_psd_gaussian(1.0, length).astype(np.float32)
        return torch.from_numpy(silence).float() * amp

    @staticmethod
    def _safe_reflect_pad(x: np.ndarray, left: int, right: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if len(x) <= 1:
            return np.pad(x, (left, right), mode="constant")
        return np.pad(x, (left, right), mode="reflect")
