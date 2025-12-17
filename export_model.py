import os
import sys
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

model_path = r"C:\Users\PhamHuynhAnh\Downloads\model_80000.pt"
output_model_path = "hpa-rmvpe.pt"

n_mels = 128
hop_length = 160
window_length = 1024
sample_rate = 16000

n_gru = 1
in_channels = 1
en_out_channels = 16

model = torch.load(model_path, map_location="cpu", weights_only=False)

new_state_dict = {}
for k, v in model["model"].items():
    new_state_dict[k.replace("module.", "")] = v

torch.save(new_state_dict, output_model_path)
print(f"Model saved to {output_model_path}")
