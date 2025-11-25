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
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    return parser.parse_args()

def process_dataset(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    df_info = pd.read_csv(args.csv_path)

    for _, row in tqdm(df_info.iterrows(), total=df_info.shape[0]):
        filename = row['name']
        label_path = row['label_path']
        split = row['split']

        wav_path_in = os.path.join(args.in_dir, filename)
        split_out_dir = os.path.join(args.out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        try:
            audio_p, _ = librosa.load(wav_path_in, sr=args.sr, mono=False)
            
            out_filename_base = os.path.splitext(os.path.basename(filename))[0]
            
            sf.write(os.path.join(split_out_dir, f"{out_filename_base}_p.wav"), 
                     audio_p.T, args.sr, 'PCM_24')

            f0_path_in = os.path.join(args.in_dir, label_path)
            if os.path.exists(f0_path_in):
                f0_hz = np.loadtxt(f0_path_in, usecols=(0,))
                f0_hz[f0_hz < 0] = 0.0
                
                pv_out_path = os.path.join(split_out_dir, f"{out_filename_base}.pv")
                np.savetxt(pv_out_path, f0_hz, fmt="%.9f")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    args = parse_args()
    process_dataset(args)