import argparse
# from .melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS
import torch
import onnx, onnxsim
import numpy as np


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
    phone_len = 256

    tts = TTS(language=language, x_len=phone_len, config_path=config_path, ckpt_path=ckpt_path, device="cpu")

    # speaker_id = 1
    
    # noise_scale = 0.6
    # duration = 3.5

    with torch.no_grad():
        phones = torch.zeros(phone_len, dtype=torch.int32)
        tones = torch.randint(1, 5, size=(phone_len,), dtype=torch.int32)
        g = torch.rand(1, 256, 1)
        lang_ids = torch.zeros(phone_len, dtype=torch.int32) + 1
        # noise_scale = torch.FloatTensor([0.667])
        # noise_scale_w = torch.FloatTensor([0.8])
        # length_scale = torch.FloatTensor([1])

        # def forward(
        #     self,
        #     x,
        #     x_lengths,
        #     tone,
        #     language,
        #     sid,
        # ):
        inputs = (
            phones, torch.IntTensor([phone_len]), g, tones, lang_ids
        )

        # Export the model
        encoder_name = "encoder.onnx"
        tts.model.forward = tts.model.enc_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                         # model input (or a tuple for multiple inputs)
                        encoder_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['x', 'x_len', 'g', 'tone', 'language'],   # the model's input names
                        output_names = ['z_p', 'y_mask', 'audio_len'], # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(encoder_name)
        onnx.save(sim_model, encoder_name)
        print(f"Save to {encoder_name}")

        # # Export the model
        # dec_len = 128
        # inputs = (
        #     torch.rand(1, 192, dec_len), torch.rand(1, 1, dec_len), g
        # )
        # flow_name = "flow.onnx"
        # tts.model.forward = tts.model.flow_forward
        # torch.onnx.export(tts.model,               # model being run
        #                 inputs,                         # model input (or a tuple for multiple inputs)
        #                 flow_name,   # where to save the model (can be a file or file-like object)
        #                 export_params=True,        # store the trained parameter weights inside the model file
        #                 opset_version=16,          # the ONNX version to export the model to
        #                 do_constant_folding=True,  # whether to execute constant folding for optimization
        #                 input_names = ['z_p', 'y_mask', 'g'],   # the model's input names
        #                 output_names = ['z'], # the model's output names
        #                 )
        # sim_model,_ = onnxsim.simplify(flow_name)
        # onnx.save(sim_model, flow_name)
        # print(f"Save to {flow_name}")

        # # Export the model
        # dec_len = 128
        # inputs = (
        #     torch.rand(1, 192, dec_len), torch.rand(1, 1, dec_len), g
        # )
        # flow_name = "flow.onnx"
        # tts.model.forward = tts.model.flow_forward
        # torch.onnx.export(tts.model,               # model being run
        #                 inputs,                         # model input (or a tuple for multiple inputs)
        #                 flow_name,   # where to save the model (can be a file or file-like object)
        #                 export_params=True,        # store the trained parameter weights inside the model file
        #                 opset_version=16,          # the ONNX version to export the model to
        #                 do_constant_folding=True,  # whether to execute constant folding for optimization
        #                 input_names = ['z_p', 'y_mask', 'g'],   # the model's input names
        #                 output_names = ['z'], # the model's output names
        #                 )
        # sim_model,_ = onnxsim.simplify(flow_name)
        # onnx.save(sim_model, flow_name)
        # print(f"Save to {flow_name}")


        # Export the model
        dec_len = 128
        inputs = (
            torch.rand(1, 192, dec_len), torch.rand(1, 1, dec_len), g
        )
        decoder_name = "flow_dec.onnx"
        tts.model.forward = tts.model.flow_dec_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                         # model input (or a tuple for multiple inputs)
                        decoder_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['z_p', 'y_mask', 'g'],   # the model's input names
                        output_names = ['audio'], # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(decoder_name)
        onnx.save(sim_model, decoder_name)
        print(f"Save to {decoder_name}")


if __name__ == "__main__":
    main()