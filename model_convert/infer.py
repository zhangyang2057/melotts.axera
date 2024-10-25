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
    phone_len = 64
    speaker_id = 1
    model = TTS(language=language, x_len=phone_len, config_path=config_path, ckpt_path=ckpt_path)
    
    output_path = f'output.wav'
    model.tts_to_file(text="道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。故常无欲，以观其妙；常有欲，以观其徼。此两者，同出而异名，同谓之玄。玄之又玄，衆妙之门", output_path=output_path)

if __name__ == "__main__":
    main()