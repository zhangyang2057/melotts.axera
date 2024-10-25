import torch
import os
from cached_path import cached_path
from huggingface_hub import hf_hub_download
import json

DOWNLOAD_CKPT_URLS = {
    'EN': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth',
    'EN_V2': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/checkpoint.pth',
    'FR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/checkpoint.pth',
    'JP': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/checkpoint.pth',
    'ES': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/checkpoint.pth',
    'ZH': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth',
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth',
}

DOWNLOAD_CONFIG_URLS = {
    'EN': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/config.json',
    'EN_V2': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/config.json',
    'FR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/config.json',
    'JP': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/config.json',
    'ES': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/config.json',
    'ZH': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json',
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/config.json',
}

PRETRAINED_MODELS = {
    'G.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/G.pth',
    'D.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/D.pth',
    'DUR.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/DUR.pth',
}

LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}

PROXIES = {
    "HTTP": "127.0.0.1:7890",
    "HTTPS": "127.0.0.1:7890"
}

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
    

def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json", proxies=PROXIES, local_dir=".")
        else:
            assert language in DOWNLOAD_CONFIG_URLS
            config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])
    return get_hparams_from_file(config_path)

def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth", proxies=PROXIES, local_dir=".")
        else:
            assert language in DOWNLOAD_CKPT_URLS
            ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])
    return torch.load(ckpt_path, map_location=device)

def load_pretrain_model():
    return [cached_path(url) for url in PRETRAINED_MODELS.values()]
