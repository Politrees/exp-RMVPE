import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
import parselmouth
import math
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--f0_min', type=float, default=80.0)
    parser.add_argument('--f0_max', type=float, default=750.0)
    return parser.parse_args()

def extract_f0_parselmouth(audio_np, sr, hop_length, f0_min, f0_max):
    mel_len = int(math.ceil(len(audio_np) / hop_length))
    time_step = hop_length / sr
    
    snd = parselmouth.Sound(audio_np, sampling_frequency=sr)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max
        )
    
    f0 = pitch.selected_array['frequency']

    delta_l = mel_len - len(f0)
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:mel_len]
    f0[np.isnan(f0)] = 0.0
    
    return f0

def process_dataset(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    df_info = pd.read_csv(args.csv_path)

    for _, row in tqdm(df_info.iterrows(), total=len(df_info)):
        relative_path = row['name']
        split = row['split']
        
        folder_name = relative_path.split('/')[0]
        file_index = os.path.splitext(relative_path.split('/')[1])[0]
        base_filename = f"{folder_name}_{file_index}"
        
        wav_path_in = os.path.join(args.in_dir, relative_path)
        split_out_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        try:
            audio, _ = librosa.load(wav_path_in, sr=args.sr, mono=False)
            
            sf.write(os.path.join(split_out_dir, f"{base_filename}_p.wav"),
                     audio.T, args.sr, 'PCM_24')
            
            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio

            f0_sequence = extract_f0_parselmouth(audio_mono, args.sr, 160, args.f0_min, args.f0_max)
            f0_sequence[f0_sequence < 0] = 0.0
            
            np.savetxt(os.path.join(split_out_dir, f"{base_filename}.pv"),
                       f0_sequence, fmt="%.6f")
                       
        except Exception as e:
            print(f"Error processing {relative_path}: {e}")

if __name__ == '__main__':
    args = parse_args()
    process_dataset(args)