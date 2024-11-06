import numpy as np
import soundfile
from axengine import InferenceSession
import onnxruntime as ort
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
    parser.add_argument("--wav", "-w", type=str, required=False, default="output.wav")
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

    enc_model = "../models/encoder.onnx"
    dec_model = "../models/decoder.axmodel"
    dec_len = 64

    # Load lexicon
    lexicon = Lexicon(lexicon_filename, token_filename)

    # Convert sentence to phones and tones
    phones, tones = lexicon.convert(sentence)

    # Add blank between words
    phones = np.array(intersperse(phones, 0), dtype=np.int32)
    tones = np.array(intersperse(tones, 0), dtype=np.int32)
    total_phone_len = phones.shape[-1]

    # Load models
    sess_enc = ort.InferenceSession(enc_model, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    sess_dec = InferenceSession.load_from_model(dec_model)

    # Load static input
    g = np.fromfile("../models/g.bin", dtype=np.float32).reshape(1,256,1)
    language = np.array([3] * total_phone_len, dtype=np.int32)

    start = time.time()
    z_p, audio_len = sess_enc.run(None, input_feed={
                                'phone': phones, 'g': g,
                                'tone': tones, 'language': language, 
                                'noise_scale': np.array([0], dtype=np.float32),
                                'length_scale': np.array([1.0], dtype=np.float32),
                                'noise_scale_w': np.array([0], dtype=np.float32),
                                'sdp_ratio': np.array([0], dtype=np.float32)})
    print(f"encoder run take {1000 * (time.time() - start)}ms")
    
    audio_len = audio_len[0]
    dec_slice_num = int(np.ceil(z_p.shape[-1] / dec_len))
    audio_list = []

    print(f"phone_len: {total_phone_len}")
    print(f"z_p.shape: {z_p.shape}")
    print(f"dec_slice_num: {dec_slice_num}")
    print(f"audio_len: {audio_len}")

    for i in range(dec_slice_num):
        z_p_slice = z_p[..., i * dec_len : (i + 1) * dec_len]
        actual_size = z_p_slice.shape[-1]

        # Pad input
        if actual_size < dec_len:
            z_p_slice = np.concatenate((z_p_slice, np.zeros((*z_p_slice.shape[:-1], dec_len - actual_size), dtype=np.float32)), axis=-1)

        start = time.time()
        audio = sess_dec.run(input_feed={"z_p": z_p_slice,
                              "g": g
                              })["audio"]
        print(f"decoder run take {1000 * (time.time() - start)}ms")

        sub_audio_len = audio.shape[-1] * (i + 1)
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
