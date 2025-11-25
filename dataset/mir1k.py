import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--pitch_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    return parser.parse_args()

def process_dataset(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    df_info = pd.read_csv(args.csv_path)
    
    for _, row in tqdm(df_info.iterrows(), total=df_info.shape[0]):
        filename = str(row.iloc[0])
        split = str(row.iloc[2])
        
        wav_path_in = os.path.join(args.wav_dir, filename)
        split_out_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        try:
            audio, _ = librosa.load(wav_path_in, sr=args.sr, mono=False)
            
            out_wav_name = filename.replace('.wav', '_m.wav')
            sf.write(os.path.join(split_out_dir, out_wav_name), audio.T, args.sr, 'PCM_24')

            pv_filename = filename.replace('.wav', '.pv')
            pv_in_path = os.path.join(args.pitch_dir, pv_filename)
            
            if os.path.exists(pv_in_path):
                f0 = np.loadtxt(pv_in_path)
                old_times = 0.020 + np.arange(len(f0)) * 0.02
                new_times = np.arange(0.020, old_times[-1] + 0.01, 0.01)
                f0_interp = np.interp(new_times, old_times, f0)
                f0_interp[np.isnan(f0_interp)] = 0.0
                np.savetxt(os.path.join(split_out_dir, pv_filename), f0_interp, fmt="%.6f")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    args = parse_args()
    process_dataset(args)