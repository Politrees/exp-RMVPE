#!/usr/bin/env python3
"""
Скрипт для подготовки датасета в формате MIR-1K-10ms для тренировки RMVPE.

Использование:
    python prepare_dataset.py --input_dir /path/to/audio --output_dir /path/to/dataset --model_path hpa-rmvpe.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
import random
from tqdm import tqdm
from typing import List, Tuple, Optional

# Добавляем путь к репозиторию
now_dir = os.getcwd()
sys.path.append(now_dir)

from src.constants import SAMPLE_RATE, WINDOW_LENGTH


# ============== Улучшенный Slicer ==============

class SilenceSlicer:
    """
    Улучшенный Slicer, который режет ТОЛЬКО по паузам между словами.
    
    Логика:
    1. Находим все участки тишины длительностью >= min_silence_duration
    2. Режем по центру этих участков
    3. Объединяем слишком короткие сегменты
    4. Разбиваем слишком длинные сегменты (ищем более короткие паузы внутри)
    """
    
    def __init__(
        self,
        sr: int = 16000,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 0.25,  # минимум 250мс тишины для разреза
        min_segment_duration: float = 1.0,   # минимальная длина сегмента
        max_segment_duration: float = 10.0,  # максимальная длина сегмента
        padding: float = 0.05,               # 50мс "подушка" с каждой стороны от точки разреза
        frame_length: int = 2048,
        hop_length: int = 512,
    ):
        self.sr = sr
        self.silence_threshold = 10 ** (silence_threshold_db / 20.0)
        self.min_silence_frames = int(min_silence_duration * sr / hop_length)
        self.min_segment_samples = int(min_segment_duration * sr)
        self.max_segment_samples = int(max_segment_duration * sr)
        self.padding_samples = int(padding * sr)
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def _get_rms(self, audio: np.ndarray) -> np.ndarray:
        """Вычисление RMS энергии для каждого фрейма."""
        # Паддинг для корректного размера
        pad_length = self.frame_length // 2
        audio_padded = np.pad(audio, (pad_length, pad_length), mode='constant')
        
        # Вычисление RMS через свёртку (быстрее чем stride tricks для больших файлов)
        n_frames = 1 + (len(audio_padded) - self.frame_length) // self.hop_length
        rms = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio_padded[start:start + self.frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        
        return rms
    
    def _find_silence_regions(self, rms: np.ndarray) -> List[Tuple[int, int]]:
        """
        Находит регионы тишины достаточной длины.
        Возвращает список (start_frame, end_frame).
        """
        is_silent = rms < self.silence_threshold
        
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                # Начало тишины
                in_silence = True
                silence_start = i
            elif not silent and in_silence:
                # Конец тишины
                in_silence = False
                silence_length = i - silence_start
                if silence_length >= self.min_silence_frames:
                    silence_regions.append((silence_start, i))
        
        # Проверяем, если файл заканчивается тишиной
        if in_silence:
            silence_length = len(is_silent) - silence_start
            if silence_length >= self.min_silence_frames:
                silence_regions.append((silence_start, len(is_silent)))
        
        return silence_regions
    
    def _frame_to_sample(self, frame_idx: int) -> int:
        """Конвертация индекса фрейма в индекс сэмпла."""
        return frame_idx * self.hop_length
    
    def _find_best_cut_point(self, rms: np.ndarray, start_frame: int, end_frame: int) -> int:
        """
        Находит лучшую точку разреза в заданном диапазоне.
        Возвращает индекс фрейма с минимальным RMS.
        """
        if start_frame >= end_frame:
            return start_frame
        region_rms = rms[start_frame:end_frame]
        best_offset = np.argmin(region_rms)
        return start_frame + best_offset
    
    def slice(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Нарезка аудио по паузам.
        
        Args:
            audio: одномерный numpy array с аудио
            
        Returns:
            Список сегментов (numpy arrays)
        """
        # Если аудио слишком короткое, возвращаем как есть
        if len(audio) <= self.min_segment_samples:
            return [audio]
        
        # Получаем RMS
        rms = self._get_rms(audio)
        
        # Находим регионы тишины
        silence_regions = self._find_silence_regions(rms)
        
        if not silence_regions:
            # Нет пауз - пробуем найти с более низким порогом
            return self._fallback_slice(audio, rms)
        
        # Определяем точки разреза (центры регионов тишины)
        cut_points_frames = []
        for start_f, end_f in silence_regions:
            # Находим точку с минимальным RMS в регионе тишины
            best_frame = self._find_best_cut_point(rms, start_f, end_f)
            cut_points_frames.append(best_frame)
        
        # Конвертируем в сэмплы
        cut_points = [self._frame_to_sample(f) for f in cut_points_frames]
        
        # Формируем сегменты
        segments = self._form_segments(audio, cut_points)
        
        return segments
    
    def _form_segments(self, audio: np.ndarray, cut_points: List[int]) -> List[np.ndarray]:
        """
        Формирует сегменты из точек разреза.
        Объединяет короткие, разбивает длинные.
        """
        # Добавляем начало и конец
        all_points = [0] + sorted(cut_points) + [len(audio)]
        
        # Формируем предварительные сегменты
        raw_segments = []
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            
            # Добавляем padding, но не выходим за границы
            if i > 0:
                start = max(0, start - self.padding_samples)
            if i < len(all_points) - 2:
                end = min(len(audio), end + self.padding_samples)
            
            if end > start:
                raw_segments.append((start, end))
        
        # Объединяем короткие сегменты
        merged_segments = self._merge_short_segments(audio, raw_segments)
        
        # Разбиваем слишком длинные
        final_segments = []
        for start, end in merged_segments:
            segment = audio[start:end]
            if len(segment) > self.max_segment_samples:
                # Пробуем разбить по более коротким паузам
                sub_segments = self._split_long_segment(segment)
                final_segments.extend(sub_segments)
            elif len(segment) >= self.min_segment_samples:
                final_segments.append(segment)
        
        # Если ничего не получилось, возвращаем исходное аудио
        if not final_segments:
            return [audio]
        
        return final_segments
    
    def _merge_short_segments(
        self, 
        audio: np.ndarray, 
        segments: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Объединяет слишком короткие сегменты с соседними."""
        if not segments:
            return segments
        
        merged = []
        current_start, current_end = segments[0]
        
        for i in range(1, len(segments)):
            next_start, next_end = segments[i]
            current_length = current_end - current_start
            
            if current_length < self.min_segment_samples:
                # Текущий сегмент короткий - объединяем со следующим
                current_end = next_end
            else:
                # Текущий сегмент достаточной длины
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        # Добавляем последний сегмент
        merged.append((current_start, current_end))
        
        # Проверяем последний сегмент
        if merged and (merged[-1][1] - merged[-1][0]) < self.min_segment_samples:
            if len(merged) > 1:
                # Объединяем с предпоследним
                prev_start, prev_end = merged[-2]
                _, last_end = merged[-1]
                merged = merged[:-2] + [(prev_start, last_end)]
        
        return merged
    
    def _split_long_segment(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Разбивает длинный сегмент, ища внутри него паузы.
        Использует более низкий порог для поиска пауз.
        """
        # Пробуем найти паузы с менее строгими требованиями
        rms = self._get_rms(audio)
        
        # Ищем локальные минимумы RMS как потенциальные точки разреза
        target_length = (self.min_segment_samples + self.max_segment_samples) // 2
        
        segments = []
        current_start = 0
        
        while current_start < len(audio):
            # Определяем окно поиска
            search_end_sample = current_start + self.max_segment_samples
            
            if search_end_sample >= len(audio):
                # Последний сегмент
                segment = audio[current_start:]
                if len(segment) >= self.min_segment_samples // 2:  # Более мягкое требование для последнего
                    segments.append(segment)
                elif segments:
                    # Добавляем к предыдущему
                    segments[-1] = np.concatenate([segments[-1], segment])
                break
            
            # Ищем лучшую точку разреза в диапазоне [target_length, max_length]
            search_start_sample = current_start + int(self.min_segment_samples * 0.8)
            search_start_frame = search_start_sample // self.hop_length
            search_end_frame = min(search_end_sample // self.hop_length, len(rms))
            
            if search_start_frame < search_end_frame:
                best_frame = self._find_best_cut_point(rms, search_start_frame, search_end_frame)
                cut_point = self._frame_to_sample(best_frame)
            else:
                cut_point = search_end_sample
            
            segment = audio[current_start:cut_point]
            if len(segment) >= self.min_segment_samples // 2:
                segments.append(segment)
            
            current_start = cut_point
        
        return segments if segments else [audio]
    
    def _fallback_slice(self, audio: np.ndarray, rms: np.ndarray) -> List[np.ndarray]:
        """
        Резервный метод: ищем точки с минимальным RMS в нужных позициях.
        Используется когда не найдено явных пауз.
        """
        if len(audio) <= self.max_segment_samples:
            return [audio]
        
        segments = []
        current_start = 0
        
        while current_start < len(audio):
            if current_start + self.max_segment_samples >= len(audio):
                segment = audio[current_start:]
                if len(segment) >= self.min_segment_samples // 2:
                    segments.append(segment)
                elif segments:
                    segments[-1] = np.concatenate([segments[-1], segment])
                break
            
            # Ищем минимум RMS в допустимом диапазоне
            search_start = current_start + self.min_segment_samples
            search_end = current_start + self.max_segment_samples
            
            search_start_frame = search_start // self.hop_length
            search_end_frame = min(search_end // self.hop_length, len(rms))
            
            best_frame = self._find_best_cut_point(rms, search_start_frame, search_end_frame)
            cut_point = self._frame_to_sample(best_frame)
            
            segments.append(audio[current_start:cut_point])
            current_start = cut_point
        
        return segments if segments else [audio]


# ============== Функции конвертации ==============

def hz_to_midi(freq: np.ndarray) -> np.ndarray:
    """Конвертация Hz в MIDI ноты. 0 Hz -> 0 (unvoiced)."""
    midi = np.zeros_like(freq)
    voiced_mask = freq > 0
    midi[voiced_mask] = 69 + 12 * np.log2(freq[voiced_mask] / 440.0)
    return midi


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """Конвертация MIDI в Hz. 0 MIDI -> 0 Hz."""
    freq = np.zeros_like(midi)
    voiced_mask = midi > 0
    freq[voiced_mask] = 440.0 * (2.0 ** ((midi[voiced_mask] - 69.0) / 12.0))
    return freq


# ============== Основной класс ==============

class DatasetPreparer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        hop_length: int = 160,
        min_segment_length: float = 1.0,
        max_segment_length: float = 10.0,
        pitch_threshold: float = 0.03,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 0.25,  # минимум 250мс паузы для разреза
        test_ratio: float = 0.1,
    ):
        self.hop_length = hop_length
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.pitch_threshold = pitch_threshold
        self.test_ratio = test_ratio
        self.device = device
        
        # Инициализация улучшенного Slicer
        self.slicer = SilenceSlicer(
            sr=SAMPLE_RATE,
            silence_threshold_db=silence_threshold_db,
            min_silence_duration=min_silence_duration,
            min_segment_duration=min_segment_length,
            max_segment_duration=max_segment_length,
        )
        
        # Загрузка модели для извлечения F0
        print(f"Загрузка модели из {model_path}...")
        self.f0_extractor = self._load_f0_model(model_path)
        print("Модель загружена!")

    def _load_f0_model(self, model_path: str):
        """Загрузка HPA-RMVPE или RMVPE модели."""
        from inference import HPARMVPE
        return HPARMVPE(model_path, device=self.device, hop_length=self.hop_length)

    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Извлечение F0 из аудио."""
        f0 = self.f0_extractor.infer_from_audio(audio, thred=self.pitch_threshold)
        return f0

    def process_audio_file(self, audio_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Обработка одного аудио файла.
        Возвращает список кортежей (audio_segment, midi_values).
        """
        # Загрузка аудио
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        if len(audio) < SAMPLE_RATE * self.min_segment_length:
            print(f"  Пропуск: слишком короткий файл ({len(audio)/SAMPLE_RATE:.2f}s)")
            return []
        
        # Нарезка на сегменты по паузам
        segments = self.slicer.slice(audio)
        
        results = []
        for seg in segments:
            # Пропуск слишком коротких сегментов
            if len(seg) < SAMPLE_RATE * self.min_segment_length * 0.5:  # Мягкий порог
                continue
            
            # Извлечение F0
            f0_hz = self.extract_f0(seg)
            
            # Проверка: есть ли вообще voiced фреймы?
            voiced_ratio = np.sum(f0_hz > 0) / len(f0_hz) if len(f0_hz) > 0 else 0
            if voiced_ratio < 0.1:  # менее 10% voiced - пропускаем
                continue
            
            # Конвертация Hz -> MIDI
            midi_values = hz_to_midi(f0_hz)
            
            results.append((seg, midi_values))
        
        return results

    def save_segment(
        self,
        output_dir: str,
        name: str,
        audio: np.ndarray,
        midi_values: np.ndarray,
    ):
        """Сохранение сегмента в формате MIR-1K."""
        wav_path = os.path.join(output_dir, f"{name}_m.wav")
        sf.write(wav_path, audio, SAMPLE_RATE)
        
        pv_path = os.path.join(output_dir, f"{name}.pv")
        with open(pv_path, 'w') as f:
            for midi in midi_values:
                f.write(f"{midi:.6f}\n")

    def prepare_dataset(
        self,
        input_dir: str,
        output_dir: str,
        audio_extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a'),
    ):
        """Основная функция подготовки датасета."""
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Поиск всех аудио файлов
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).rglob(f'*{ext}'))
            audio_files.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
        
        audio_files = sorted(set(audio_files))
        print(f"Найдено {len(audio_files)} аудио файлов")
        
        if not audio_files:
            print("Аудио файлы не найдены!")
            return
        
        # Обработка всех файлов
        all_segments = []
        
        for audio_path in tqdm(audio_files, desc="Обработка файлов"):
            try:
                segments = self.process_audio_file(str(audio_path))
                source_name = Path(audio_path).stem
                for i, (audio, midi) in enumerate(segments):
                    all_segments.append((audio, midi, f"{source_name}_{i:04d}"))
            except Exception as e:
                print(f"\nОшибка при обработке {audio_path}: {e}")
                continue
        
        print(f"\nВсего получено {len(all_segments)} сегментов")
        
        if not all_segments:
            print("Не удалось создать ни одного сегмента!")
            return
        
        # Перемешивание и разбиение на train/test
        random.shuffle(all_segments)
        n_test = max(1, int(len(all_segments) * self.test_ratio))
        n_train = len(all_segments) - n_test
        
        train_segments = all_segments[:n_train]
        test_segments = all_segments[n_train:]
        
        print(f"Train: {len(train_segments)} сегментов")
        print(f"Test: {len(test_segments)} сегментов")
        
        # Сохранение
        print("\nСохранение train...")
        for audio, midi, name in tqdm(train_segments):
            self.save_segment(train_dir, name, audio, midi)
        
        print("Сохранение test...")
        for audio, midi, name in tqdm(test_segments):
            self.save_segment(test_dir, name, audio, midi)
        
        self._print_statistics(train_segments, test_segments)
        print(f"\nДатасет сохранён в {output_dir}")

    def _print_statistics(self, train_segments, test_segments):
        """Вывод статистики датасета."""
        def calc_stats(segments):
            if not segments:
                return 0, 0, 0
            total_duration = sum(len(a) / SAMPLE_RATE for a, _, _ in segments)
            voiced_frames = sum(np.sum(m > 0) for _, m, _ in segments)
            total_frames = sum(len(m) for _, m, _ in segments)
            return total_duration, voiced_frames, total_frames
        
        train_dur, train_voiced, train_total = calc_stats(train_segments)
        test_dur, test_voiced, test_total = calc_stats(test_segments)
        
        print("\n" + "="*50)
        print("СТАТИСТИКА ДАТАСЕТА")
        print("="*50)
        print(f"Train:")
        print(f"  Сегментов: {len(train_segments)}")
        print(f"  Общая длительность: {train_dur:.1f}с ({train_dur/60:.1f} мин)")
        if train_total > 0:
            print(f"  Voiced фреймов: {train_voiced}/{train_total} ({100*train_voiced/train_total:.1f}%)")
        print(f"Test:")
        print(f"  Сегментов: {len(test_segments)}")
        print(f"  Общая длительность: {test_dur:.1f}с ({test_dur/60:.1f} мин)")
        if test_total > 0:
            print(f"  Voiced фреймов: {test_voiced}/{test_total} ({100*test_voiced/test_total:.1f}%)")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета для тренировки RMVPE")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Директория с исходными аудио файлами")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Директория для сохранения датасета")
    parser.add_argument("--model_path", "-m", type=str, default="hpa-rmvpe.pt",
                        help="Путь к модели HPA-RMVPE")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Устройство для инференса (cuda/cpu)")
    parser.add_argument("--min_length", type=float, default=1.0,
                        help="Минимальная длина сегмента в секундах (default: 1.0)")
    parser.add_argument("--max_length", type=float, default=10.0,
                        help="Максимальная длина сегмента в секундах (default: 10.0)")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="Порог pitch для voiced/unvoiced (default: 0.03)")
    parser.add_argument("--silence_threshold", type=float, default=-40.0,
                        help="Порог тишины в dB (default: -40.0)")
    parser.add_argument("--min_silence", type=float, default=0.25,
                        help="Минимальная длительность паузы для разреза в секундах (default: 0.25)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Доля данных для test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed для воспроизводимости")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    preparer = DatasetPreparer(
        model_path=args.model_path,
        device=args.device,
        min_segment_length=args.min_length,
        max_segment_length=args.max_length,
        pitch_threshold=args.threshold,
        silence_threshold_db=args.silence_threshold,
        min_silence_duration=args.min_silence,
        test_ratio=args.test_ratio,
    )
    
    preparer.prepare_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()