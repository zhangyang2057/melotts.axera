import numpy as np
import onnxruntime as ort
import soundfile
from axengine import InferenceSession
import argparse


def get_argparser():
    parser = argparse.ArgumentParser(
        prog="onnx_infer_stream",
        description="Run TTS on input sentence"
    )
    parser.add_argument("--sentence", "-s", type=str, required=False, default="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用")
    parser.add_argument("--speed", type=float, required=False, default=1.0)
    parser.add_argument("--sample_rate", "-sr", type=int, required=False, default=44100)
    return parser


def load_pinyin_dict(file_path="../models/melo_lexicon_zh.txt"):
    '''
    读取发音字典 格式为
    单词#phone1 空格 phone2#tone1 空格 tone2
    '''
    pinyin_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('#')
            if len(parts) > 1:
                character = parts[0]
                pinyin_parts = parts[1].split()
                tone = parts[-1]
                pinyin_dict[character] = (pinyin_parts, tone)
    return pinyin_dict


def get_pinyin_and_tones(sentence, pinyin_dict):
    '''
    输入句子和发音字典
    返回对应的拼音列表和音调列表
    '''
    pinyin_list = []
    tone_list = []
    for char in sentence:
        if char in pinyin_dict:
            pinyin, tone = pinyin_dict[char]
            pinyin_list.append(pinyin)
            tone_list.append(tone)
        else:
            pinyin_list.append(char)
            tone_list.append('')  # 若找不到对应拼音，音调留空

    pinyin_list = [item for sublist in pinyin_list for item in sublist]
    tone_list = [item for sublist in tone_list for item in sublist]
    tone_list_ = []
    for x in tone_list:
        tones = x.split()
        for xx in tones:
            tone_list_.append(int(xx))
    tone_list = tone_list_
    return pinyin_list, tone_list


def load_syllable_dict(file_path="../models/melo_tokens.txt"):
    syllable_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                syllable = parts[0]
                number = parts[1]
                syllable_dict[syllable] = int(number)
    return syllable_dict


def replace_syllables_with_numbers(array, syllable_dict):
    return [syllable_dict.get(item, item) for item in array]


def insert_zeros(arr):
    '''
    输入 1 2 3
    输出0 1 0 2 0 3 0
    '''
    # 计算插入0后新数组的长度
    new_length = len(arr) * 2
    # 创建新数组，初始值全部为0
    new_arr = np.zeros(new_length, dtype=arr.dtype)
    # 将原始数组的值按顺序插入到新数组中
    new_arr[1::2] = arr
    new_arr = np.append(new_arr,0)
    return new_arr


def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def main():
    args = get_argparser().parse_args()
    # 输入句子
    sentence = args.sentence
    print(f"sentence: {sentence}")
    sentence += "........"

    speed = args.speed
    sample_rate = args.sample_rate

    print(f"speed: {speed}")
    print(f"sample_rate: {sample_rate}")

    # 加载拼音字典
    pinyin_dict = load_pinyin_dict()

    
    
    # 获取拼音和音调列表
    pinyin_list, tone_list = get_pinyin_and_tones(sentence, pinyin_dict)

    print("拼音列表:", pinyin_list)
    print("音调列表:", tone_list)


    syllable_dict = load_syllable_dict()

    # 替换音节为对应数字
    replaced_array = replace_syllables_with_numbers(pinyin_list, syllable_dict)

    # print("替换后的数组:", replaced_array)

    replaced_array = np.pad(replaced_array, pad_width=1, mode='constant', constant_values=0)
    tone_array = np.pad(tone_list, pad_width=1, mode='constant', constant_values=0)
    langids = np.zeros_like(tone_array) + 3

    # print("pad phone:", replaced_array)
    # print("pad tone:", tone_array)
    # print("langids:", langids)


    x_tst = insert_zeros(replaced_array).reshape((1,-1))
    xlen = len(x_tst[0])
    tones = insert_zeros(tone_array).reshape((1,-1))
    langids = insert_zeros(langids).reshape((1,-1))

    # print("insert_zeros phone:", x_tst)
    # print("insert_zeros tone:", tones)
    # print("insert_zeros langids:", langids)

    x_tst = np.array(x_tst,dtype=np.int64)
    xlen = np.array([xlen],dtype=np.int64)
    tones = np.array(tones,dtype=np.int64)

    langids = np.zeros_like(x_tst) + 3
    langids =np.array(langids,dtype=np.int64)
    bert = np.zeros((1,1024,len(x_tst[0])),dtype=np.float32)
    jabert = np.zeros((1,768,len(x_tst[0])),dtype=np.float32)


    #-------------------------------------推理enc
    onnx_model = "../models/enc-sim.onnx"

    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(
        onnx_model, providers=providers, sess_options=sess_options)

    g = np.fromfile('../models/g.bin',dtype=np.float32).reshape(1,256,1)
    x = sess.run(None, input_feed={'x_tst': x_tst, 'x_tst_l': xlen, 'g': g,
                                'tones': tones, 'langids': langids, 'bert': bert, 'jabert': jabert,
                                'noise_scale': np.array([0.3], dtype=np.float32),
                                'length_scale': np.array([1.0 / speed], dtype=np.float32),
                                'noise_scale_w': np.array([0.8], dtype=np.float32),
                                'sdp_ratio': np.array([0.2], dtype=np.float32)})

    zp,ymask = x[0],x[1] # zp 1 192 mellen  ymask 1 1 mellen 全为1



    #-------------------------------------推理dec和flow
    dec_model = "../models/decoder.axmodel"
    flow_model = "../models/flow.axmodel"

    sessd = InferenceSession()
    sessd.load_model(dec_model)

    sessf = InferenceSession()
    sessf.load_model(flow_model)

    mellen = ymask.shape[2]
    segsize = 120
    padsize = 10

    ymaskseg = np.ones((1,1,segsize),dtype=np.float32)
    wavlist = []
    i = 0

    ymaskseg = ymaskseg.flatten()
    g = g.flatten()
    while(i+segsize<=mellen):
        segz = zp[:,:,i:i+segsize]
        i += segsize-2*padsize
        
        segz = segz.flatten()
        sessf.feed_inputs([segz, ymaskseg, g])
        sessf.forward()
        x = sessf.get_outputs(["6797"])
        flowout = x["6797"].flatten()

        sessd.feed_inputs([flowout, g])
        sessd.forward()
        x = sessd.get_outputs(["827"])

        wav = np.array(x["827"], dtype=np.float32).flatten()
        wav *= 5
        # wavlist.append(wav[padsize*512:-padsize*512])
        wavlist.append(wav)

    # wavlist = np.array(wavlist).flatten()
    wavlist = audio_numpy_concat(wavlist, sr=sample_rate, speed=speed)    
    outfile = "./test_cn.wav"
    soundfile.write(outfile, wavlist, sample_rate)
    print(f"Save to {outfile}")


if __name__ == "__main__":
    main()
