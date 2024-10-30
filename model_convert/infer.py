import argparse
from melotts.download_utils import load_or_download_config, load_or_download_model
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
    phone_len = 256
    speaker_id = 1
    if not os.path.exists(config_path):
        load_or_download_config(locale="ZH")
    if not os.path.exists(ckpt_path):
        load_or_download_model(locale="ZH", device=device)
    model = TTS(language=language, x_len=phone_len, config_path=config_path, ckpt_path=ckpt_path, device=device)
    
    output_path = f'output.wav'
    model.tts_to_file(text="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用", speaker_id=speaker_id, output_path=output_path, save_data=True)

if __name__ == "__main__":
    main()