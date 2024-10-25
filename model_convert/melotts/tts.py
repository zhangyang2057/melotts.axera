import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.nn.functional as F

from .models import SynthesizerTrn
from .split_utils import split_sentence
from .download_utils import load_or_download_config, load_or_download_model
from .text import cleaned_text_to_sequence, get_bert
from .text.cleaner import clean_text
from .commons import intersperse

def get_text_for_tts_infer(text, language_str, hps, device, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = get_bert(norm_text, word2ph, language_str, device)
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


class TTS(nn.Module):
    def __init__(self, 
                language,
                x_len=128,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            x_len=x_len,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        # texts = self.split_sentences_into_pieces(text, language, quiet)
        texts = [text]
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            
            # slice 到固定长度
            phone_len = self.model.x_len
            bert = torch.zeros(1024, phone_len)
            ja_bert = torch.zeros(768, phone_len)
            nslice = int(np.ceil(phones.size(0) * 1.0 / phone_len))
            print(f"phones.size() = {phones.size(0)}")
            with torch.no_grad():
                for i in range(nslice):
                    phones_slice = phones[i * phone_len: (i+1) * phone_len]
                    tones_slice = tones[i * phone_len: (i+1) * phone_len]
                    langids_slice = lang_ids[i * phone_len: (i+1) * phone_len]
                    slice_len = phones_slice.size(0)
                    if slice_len < phone_len:
                        phones_slice = F.pad(phones_slice, (0, phone_len - slice_len), mode="constant", value=0)
                        tones_slice = F.pad(tones_slice, (0, phone_len - slice_len), mode="constant", value=0)
                        langids_slice = F.pad(langids_slice, (0, phone_len - slice_len), mode="constant", value=0)

                    print(f"phones_slice: {phones_slice.size()}")
                    print(f"tones_slice: {tones_slice.size()}")
                    print(f"langids_slice: {langids_slice.size()}")

                    # x_tst = phones.to(device).unsqueeze(0)
                    # tones = tones.to(device).unsqueeze(0)
                    # lang_ids = lang_ids.to(device).unsqueeze(0)
                    phones_slice = phones_slice.to(device).unsqueeze(0)
                    tones_slice = tones_slice.to(device).unsqueeze(0)
                    langids_slice = langids_slice.to(device).unsqueeze(0)
                    # bert = bert.to(device).unsqueeze(0)
                    # ja_bert = ja_bert.to(device).unsqueeze(0)
                    # x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                    x_tst_lengths = torch.LongTensor([slice_len]).to(device)
                    # del phones
                    speakers = torch.LongTensor([speaker_id]).to(device)
                    audio, audio_size = self.model.forward(
                            phones_slice,
                            x_tst_lengths,
                            tones_slice,
                            langids_slice,
                            speakers
                        )
                    audio = audio[0, 0].data.cpu().float().numpy()
                    audio_size = audio_size.item()
                    # del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                        # 
                    audio = audio[:audio_size]
                    audio_list.append(audio)
                    print(f"audio.size: {audio.shape[-1]}")
                    
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
