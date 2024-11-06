import argparse
from .melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS
import torch
import onnx, onnxsim
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="ZH",
        choices=["EN", "FR", "JP", "ES", "ZH", "KR"],
        help="target language for TTS",
    )

    args = parser.parse_args()
    return args


def main():
    language = get_args().language
    config_path = "config.json"
    ckpt_path = "checkpoint.pth"
    device = "cpu"
    if not os.path.exists(config_path):
        load_or_download_config(locale=language)
    if not os.path.exists(ckpt_path):
        load_or_download_model(locale=language, device=device)

    tts = TTS(language=language, dec_len=128, config_path=config_path, ckpt_path=ckpt_path, device=device)

    phone_len = 256
    with torch.no_grad():
        phones = torch.zeros(phone_len, dtype=torch.int32)
        tones = torch.randint(1, 5, size=(phone_len,), dtype=torch.int32)
        g = torch.rand(1, 256, 1)
        lang_ids = torch.zeros(phone_len, dtype=torch.int32) + 3
        noise_scale = torch.FloatTensor([0.667])
        noise_scale_w = torch.FloatTensor([0.8])
        length_scale = torch.FloatTensor([1])
        sdp_ratio = torch.FloatTensor([0])

        inputs = (
            phones, tones, lang_ids, g, noise_scale, noise_scale_w, length_scale, sdp_ratio
        )
        input_names = ['phone','tone', 'language', 'g', 'noise_scale', 'noise_scale_w', 'length_scale', 'sdp_ratio']
        dynamic_axes = {
            "phone": {0: "phone_len"},
            "tone": {0: "phone_len"},
            "language": {0: "phone_len"},
        }
        # Export the model
        encoder_name = "encoder.onnx"
        tts.model.forward = tts.model.enc_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                         # model input (or a tuple for multiple inputs)
                        encoder_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        dynamic_axes=dynamic_axes,
                        input_names = input_names,   # the model's input names
                        output_names = ['z_p', 'audio_len'], # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(encoder_name)
        onnx.save(sim_model, encoder_name)
        print(f"Save to {encoder_name}")

        # Export the model
        dec_len = tts.model.dec_len
        inputs = (
            torch.rand(1, 192, dec_len), g
        )
        decoder_name = "decoder.onnx"
        tts.model.forward = tts.model.flow_dec_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                         # model input (or a tuple for multiple inputs)
                        decoder_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['z_p', 'g'],   # the model's input names
                        output_names = ['audio'], # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(decoder_name)
        onnx.save(sim_model, decoder_name)
        print(f"Save to {decoder_name}")


if __name__ == "__main__":
    main()