import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import soundfile
import onnxruntime as ort
import argparse
import time
from utils import *
from split_utils import split_sentence
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from symbols import *

def get_text_for_tts_infer(text, language_str, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    yinjie_num, phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)

    return yinjie_num, phone, tone, language

def split_sentences_into_pieces(text, language, quiet=False):
    texts = split_sentence(text, language_str=language)
    if not quiet:
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
    return texts

def get_args():
    parser = argparse.ArgumentParser(
        prog="melotts",
        description="Run TTS on input sentence"
    )
    parser.add_argument("--sentence", "-s", type=str, required=False, default="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用")
    parser.add_argument("--wav", "-w", type=str, required=False, default="output.wav")
    parser.add_argument("--encoder", "-e", type=str, required=False, default="../models/encoder.onnx")
    parser.add_argument("--decoder", "-d", type=str, required=False, default="../models/decoder.onnx")
    parser.add_argument("--sample_rate", "-sr", type=int, required=False, default=44100)
    parser.add_argument("--speed", type=float, required=False, default=0.8)
    parser.add_argument("--lexicon", type=str, required=False, default="../models/lexicon.txt")
    parser.add_argument("--token", type=str, required=False, default="../models/tokens.txt")
    parser.add_argument("--language", type=str, 
                        choices=["ZH", "ZH_MIX_EN", "JP", "EN", 'KR', "ES", "SP","FR"], required=False, default="ZH_MIX_EN")
    return parser.parse_args()


def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    # Average pad part
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]

    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]


def main():
    args = get_args()
    sentence = args.sentence
    sample_rate = args.sample_rate
    lexicon_filename = args.lexicon
    token_filename = args.token
    enc_model = args.encoder # default="../models/encoder.onnx"
    dec_model = args.decoder # default="../models/decoder.onnx"
    language = args.language # default: ZH_MIX_EN

    print(f"sentence: {sentence}")
    print(f"sample_rate: {sample_rate}")
    print(f"lexicon: {lexicon_filename}")
    print(f"token: {token_filename}")
    print(f"encoder: {enc_model}")
    print(f"decoder: {dec_model}")
    print(f"language: {language}")

    _symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[language])}

    # Split sentence
    # sens = split_sentences_zh(sentence)
    start = time.time()
    sens = split_sentences_into_pieces(sentence, language, quiet=False)
    print(f"split_sentences_into_pieces take {1000 * (time.time() - start)}ms")

    # Load lexicon
    lexicon = Lexicon(lexicon_filename, token_filename)

    # Load models
    start = time.time()
    sess_enc = ort.InferenceSession(enc_model, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    sess_dec = ort.InferenceSession(dec_model, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    dec_len = 65536 // 512
    print(f"load models take {1000 * (time.time() - start)}ms")

    # Load static input
    g = np.fromfile("../models/g.bin", dtype=np.float32).reshape(1, 256, 1)

    # Final wav
    audio_list = []

    # Iterate over splitted sentences
    for n, se in enumerate(sens):
        if language in ['EN', 'ZH_MIX_EN']:
            se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)
        print(f"\nSentence[{n}]: {se}")
        # Convert sentence to phones and tones
        # phone_str, yinjie_num, phones, tones = lexicon.convert(se)
        yinjie_num, phones, tones, lang_ids = get_text_for_tts_infer(se, language, symbol_to_id=_symbol_to_id)

        # Add blank between words
        phone_str = intersperse(se, 0)
        phones = np.array(phones, dtype=np.int32)
        tones = np.array(tones, dtype=np.int32)
        
        phone_len = phones.shape[-1]

        start = time.time()
        z_p, pronoun_lens, audio_len = sess_enc.run(None, input_feed={
                                    'phone': phones, 'g': g,
                                    'tone': tones, 'language': lang_ids, 
                                    'noise_scale': np.array([0], dtype=np.float32),
                                    'length_scale': np.array([1.0 / args.speed], dtype=np.float32),
                                    'noise_scale_w': np.array([0], dtype=np.float32),
                                    'sdp_ratio': np.array([0], dtype=np.float32)})
        print(f"encoder run take {1000 * (time.time() - start):.2f}ms")
        
        audio_len = audio_len[0]
        actual_size = z_p.shape[-1]
        dec_slice_num = int(np.ceil(actual_size / dec_len))
        # print(f"origin z_p.shape: {z_p.shape}")
        z_p = np.pad(z_p, pad_width=((0,0),(0,0),(0, dec_slice_num * dec_len - actual_size)), mode="constant", constant_values=0)

        sub_audio_list = []
        for i in range(dec_slice_num):
            z_p_slice = z_p[..., i * dec_len : (i + 1) * dec_len]
            sub_dec_len = z_p_slice.shape[-1]
            sub_audio_len = 512 * sub_dec_len

            if z_p_slice.shape[-1] < dec_len:
                z_p_slice = np.concatenate((z_p_slice, np.zeros((*z_p_slice.shape[:-1], dec_len - z_p_slice.shape[-1]), dtype=np.float32)), axis=-1)

            start = time.time()
            audio = sess_dec.run(None, input_feed={"z_p": z_p_slice,
                                "g": g
                                })[0].flatten()
            audio = audio[:sub_audio_len]
            print(f"Long word slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")
            sub_audio_list.append(audio)
        sub_audio = merge_sub_audio(sub_audio_list, 0, audio_len)
        audio_list.append(sub_audio)

    audio = audio_numpy_concat(audio_list, sr=sample_rate, speed=args.speed)
    soundfile.write(args.wav, audio, sample_rate)
    print(f"Save to {args.wav}")

if __name__ == "__main__":
    main()
