import os
import sys
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

model_path = r"C:\Users\PhamHuynhAnh\Downloads\model_80000.pt" # "model-76000.pt"
# output model hpa-rmvpe pytorch
output_model_path = "hpa-rmvpe.pt"
output_model_fp16_path = "hpa-rmvpe-fp16.pt"
# output model hpa rmvpe onnx
output_model_onnx_path = "hpa-rmvpe.onnx"
output_model_onnx_fp16_path = "hpa-rmvpe-fp16.onnx"

export_onnx = True
pytorch_fp16 = True
onnxruntime_fp16 = True

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

if pytorch_fp16:
    fp16_new_state_dict = {}
    for key in new_state_dict:
        fp16_new_state_dict[key] = new_state_dict[key].half()
    
    torch.save(fp16_new_state_dict, output_model_fp16_path)

if export_onnx:
    import onnx
    import onnxsim
    import onnxconverter_common

    from src.spec import MelSpectrogram
    from src.model import E2E0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mel_extractor = MelSpectrogram(n_mels, sample_rate, window_length, hop_length, None, 30, sample_rate // 2).to(device)

    waveform = torch.randn(1, 114514, dtype=torch.float32, device=device).clip(min=-1., max=1.)
    mel = mel_extractor(waveform, center=True)

    model = E2E0(n_gru, in_channels, en_out_channels)
    model.load_state_dict(new_state_dict)
    model = model.to(device).eval()

    n_frames = mel.shape[-1]
    mel = torch.nn.functional.pad(
        mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
    )

    torch.onnx.export(
        model,
        (
            mel,
        ),
        output_model_onnx_path,
        do_constant_folding=True, 
        verbose=False, 
        input_names=[
            'mel',
        ],
        output_names=[
            'f0'
        ],
        dynamic_axes={
            'mel': [2],
            'f0': [1]
        },
    )

    model, _ = onnxsim.simplify(output_model_onnx_path)
    onnx.save(model, output_model_onnx_path)

    # Convert model to float16 arithmetic
    if onnxruntime_fp16:
        convert_model = onnxconverter_common.convert_float_to_float16(
            onnx.load(output_model_onnx_path), 
            keep_io_types=True
        )

        onnx.save(convert_model, output_model_onnx_fp16_path)