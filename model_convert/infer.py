import argparse
# from .melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS
import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    
    save_path = f'output.wav'
    model.tts_to_file("顶你个肺", 1, save_path)

if __name__ == "__main__":
    main()