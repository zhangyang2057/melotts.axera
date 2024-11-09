import re
from typing import Iterable, List, Tuple
import cn2an


def merge_short_sentences_zh(sens):
    # return sens
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out


def split_sentences_zh(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    # 将文本中的换行符、空格和制表符替换为空格
    text = re.sub('[\n\t ]+', ' ', text)
    # 在标点符号后添加一个空格
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    # 分隔句子并去除前后空格
    # sentences = [s.strip() for s in re.split('(。|！|？|；)', text)]
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def replace_numbers(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


def replace_punctuation(text):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "$": ".",
        "“": "'",
        "”": "'",
        "‘": "'",
        "’": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
    }

    for k, v in rep_map.items():
        text = text.replace(k, v)
    return text


def text_normalize(text):
    text = replace_numbers(text)
    text = replace_punctuation(text)
    return text


class Lexicon:
    def __init__(self, lexion_filename: str, tokens_filename: str):
        tokens = dict()
        with open(tokens_filename, encoding="utf-8") as f:
            for line in f:
                s, i = line.split()
                tokens[s] = int(i)

        lexicon = dict()
        with open(lexion_filename, encoding="utf-8") as f:
            for line in f:
                splits = line.split()
                word_or_phrase = splits[0]
                phone_tone_list = splits[1:]
                assert len(phone_tone_list) & 1 == 0, len(phone_tone_list)
                phone_str = phone_tone_list[: len(phone_tone_list) // 2]
                phones = [tokens[p] for p in phone_str]

                tones = phone_tone_list[len(phone_tone_list) // 2 :]
                tones = [int(t) for t in tones]

                lexicon[word_or_phrase] = (phone_str, phones, tones)
        lexicon["呣"] = lexicon["母"]
        lexicon["嗯"] = lexicon["恩"]
        self.lexicon = lexicon

        punctuation = ["!", "?", "…", ",", ".", "'", "-"]
        for p in punctuation:
            i = tokens[p]
            tone = 0
            self.lexicon[p] = ([p], [i], [tone])
        self.lexicon[" "] = ([" "], [tokens["_"]], [0])

    def g2p_zh_mix_en(self, text: str) -> Tuple[List[int], List[int]]:
        phone_str = []
        phones = []
        tones = []

        if text not in self.lexicon:
            # print("t", text)
            if len(text) > 1:
                for w in text:
                    # print("w", w)
                    s, p, t = self.convert(w)
                    if p:
                        phone_str += s
                        phones += p
                        tones += t
            return phone_str, phones, tones

        phone_str, phones, tones = self.lexicon[text]
        return phone_str, phones, tones
    
    
    def split_zh_en(self, text):
        spliter = '#$&^!@'
        # replace all english words
        text = re.sub('([a-zA-Z\s]+)', lambda x: f'{spliter}{x.group(1)}{spliter}', text)
        texts = text.split(spliter)
        texts = [t for t in texts if len(t) > 0]
        return texts
    

    def convert(self, text_list: Iterable[str]) -> Tuple[List[int], List[int]]:
        phone_str = []
        yinjie_num = []
        phones = []
        tones = []
        for text in text_list:
            texts_zh_en = self.split_zh_en(text)
            for text_one_lang in texts_zh_en:
                s, p, t = self.g2p_zh_mix_en(text_one_lang)

                phone_str += s
                yinjie_num.append(len(s))
                phones += p
                tones += t
        return phone_str, yinjie_num, phones, tones