import numpy as np
import soundfile
from axengine import InferenceSession
import argparse
import time
from utils import *
import math


def get_args():
    parser = argparse.ArgumentParser(
        prog="melotts",
        description="Run TTS on input sentence"
    )
    parser.add_argument("--sentence", "-s", type=str, required=False, default="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用")
    parser.add_argument("--wav", "-w", type=str, required=False, default="test_cn.wav")
    parser.add_argument("--sample_rate", "-sr", type=int, required=False, default=44100)
    parser.add_argument("--lexicon", type=str, required=False, default="../models/lexicon.txt")
    parser.add_argument("--token", type=str, required=False, default="../models/tokens.txt")
    return parser.parse_args()


def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def main():
    args = get_args()
    sentence = args.sentence
    sample_rate = args.sample_rate
    lexicon_filename = args.lexicon
    token_filename = args.token
    print(f"sentence: {sentence}")
    print(f"sample_rate: {sample_rate}")
    print(f"lexicon: {lexicon_filename}")
    print(f"token: {token_filename}")

    # Load lexicon
    lexicon = Lexicon(lexicon_filename, token_filename)

    # Convert sentence to phones and tones
    phones, tones = lexicon.convert(sentence)

    # Add blank between words
    phones = np.array(intersperse(phones, 0), dtype=np.int32)
    tones = np.array(intersperse(tones, 0), dtype=np.int32)
    total_phone_len = phones.shape[-1]

    # Load models
    enc_model = "../models/encoder.axmodel"
    dec_model = "../models/flow_dec.axmodel"

    sess_enc = InferenceSession.load_from_model(enc_model)
    sess_dec = InferenceSession.load_from_model(dec_model)

    enc_len = 256
    dec_len = 128
    enc_slice_num = int(math.ceil(total_phone_len / enc_len))

    # Load static input
    g = np.fromfile("../models/g.bin", dtype=np.float32)
    language = np.array([3] * enc_len, dtype=np.int32)

    audio_list = []
    for i in range(enc_slice_num):
        phone_slice = phones[i * enc_len : (i+1) * enc_len]
        tone_slice = tones[i * enc_len : (i+1) * enc_len]
        actual_phone_len = len(phone_slice)

        # Pad input
        if actual_phone_len < enc_len:
            phone_slice = np.append(phone_slice, np.zeros((enc_len - actual_phone_len,), dtype=np.int32))
            tone_slice = np.append(tone_slice, np.zeros((enc_len - actual_phone_len,), dtype=np.int32))

        actual_phone_len = np.array([actual_phone_len], dtype=np.int32)
        x = sess_enc.run(input_feed={"x": phone_slice, 
                          "x_len": actual_phone_len,
                          "g": g,
                          "tone": tone_slice,
                          "language": language
                          })
        z_p, y_mask, audio_len = x["z_p"], x["y_mask"], int(x["audio_len"][0])
        print(f"x: {x}")

        print(f"audio_len: {audio_len}")
        
        for n in range(y_mask.shape[-1] // dec_len):
            audio = sess_dec.run(input_feed={"z_p": z_p[..., n * dec_len : (n+1) * dec_len],
                              "y_mask": y_mask[..., n * dec_len : (n+1) * dec_len],
                              "g": g
                              })["audio"]
            sub_audio_len = audio.shape[-1] * (n + 1)
            if sub_audio_len > audio_len:
                audio = audio[:audio_len - sub_audio_len]
                audio_list.append(audio)
                break
            else:
                audio_list.append(audio)

    audio = audio_numpy_concat(audio_list, sr=sample_rate)
    soundfile.write(args.wav, audio, sample_rate)
    print(f"Save to {args.wav}")

if __name__ == "__main__":
    main()
