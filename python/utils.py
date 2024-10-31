import re
from typing import Iterable, List, Tuple


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
                phones = phone_tone_list[: len(phone_tone_list) // 2]
                phones = [tokens[p] for p in phones]

                tones = phone_tone_list[len(phone_tone_list) // 2 :]
                tones = [int(t) for t in tones]

                lexicon[word_or_phrase] = (phones, tones)
        lexicon["呣"] = lexicon["母"]
        lexicon["嗯"] = lexicon["恩"]
        self.lexicon = lexicon

        punctuation = ["!", "?", "…", ",", ".", "'", "-"]
        for p in punctuation:
            i = tokens[p]
            tone = 0
            self.lexicon[p] = ([i], [tone])
        self.lexicon[" "] = ([tokens["_"]], [0])

    def _convert(self, text: str) -> Tuple[List[int], List[int]]:
        phones = []
        tones = []

        if text == "，":
            text = ","
        elif text == "。":
            text = "."
        elif text == "！":
            text = "!"
        elif text == "？":
            text = "?"

        if text not in self.lexicon:
            # print("t", text)
            if len(text) > 1:
                for w in text:
                    # print("w", w)
                    p, t = self.convert(w)
                    if p:
                        phones += p
                        tones += t
            return phones, tones

        phones, tones = self.lexicon[text]
        return phones, tones

    def convert(self, text_list: Iterable[str]) -> Tuple[List[int], List[int]]:
        phones = []
        tones = []
        for text in text_list:
            # print(text)
            p, t = self._convert(text)
            phones += p
            tones += t
        return phones, tones