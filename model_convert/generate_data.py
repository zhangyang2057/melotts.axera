import argparse
from melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS
import torch
import os
import json

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
    speaker_id = 1
    
    if not os.path.exists(config_path):
        load_or_download_config(locale=language)
    if not os.path.exists(ckpt_path):
        load_or_download_model(locale=language, device=device)

    with open(config_path, "r") as f:
        config = json.load(f)
        speaker_id = config["data"]["spk2id"][language]

        print(f"speaker_id: {speaker_id}")

    model = TTS(language=language, dec_len=128, config_path=config_path, ckpt_path=ckpt_path, device=device)
    model.generate_data(text="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用", speaker_id=speaker_id)

if __name__ == "__main__":
    main()