/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "cmdline.hpp"
#include "OnnxWrapper.hpp"
#include <ax_sys_api.h>
#include "EngineWrapper.hpp"
#include "AudioFile.h"

typedef std::pair<std::vector<std::string>, std::vector<std::string>>    PINYIN_DICT_VAL_TYPE;
typedef std::unordered_map<std::string, PINYIN_DICT_VAL_TYPE>   PINYIN_DICT_TYPE;

std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

std::vector<std::string> splitEachChar(const std::string& chars)
{
    std::vector<std::string> words;
    std::string input(chars);
    int len = input.length();
    int i = 0;
    
    while (i < len) {
      int next = 1;
      if ((input[i] & 0x80) == 0x00) {
        // std::cout << "one character: " << input[i] << std::endl;
      } else if ((input[i] & 0xE0) == 0xC0) {
        next = 2;
        // std::cout << "two character: " << input.substr(i, next) << std::endl;
      } else if ((input[i] & 0xF0) == 0xE0) {
        next = 3;
        // std::cout << "three character: " << input.substr(i, next) << std::endl;
      } else if ((input[i] & 0xF8) == 0xF0) {
        next = 4;
        // std::cout << "four character: " << input.substr(i, next) << std::endl;
      }
      words.push_back(input.substr(i, next));
      i += next;
    }
    return words;
} 

// def load_pinyin_dict(file_path="melo_lexicon_zh.txt"):
//     '''
//     读取发音字典 格式为
//     单词#phone1 空格 phone2#tone1 空格 tone2
//     '''
//     pinyin_dict = {}
//     with open(file_path, 'r', encoding='utf-8') as file:
//         for line in file:
//             parts = line.strip().split('#')
//             if len(parts) > 1:
//                 character = parts[0]
//                 pinyin_parts = parts[1].split()
//                 tone = parts[-1]
//                 pinyin_dict[character] = (pinyin_parts, tone)
//     return pinyin_dict
static bool load_pinyin_dict(PINYIN_DICT_TYPE& pinyin_dict, const std::string& lexicon_file) {
    std::ifstream ifs(lexicon_file);
    if (!ifs.is_open()) {
        printf("Open lexicon %s failed!\n", lexicon_file.c_str());
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        auto parts = split(line, '#');
        if (parts.size() > 1) {
            std::string character = parts[0];
            auto pinyin_parts = split(parts[1], ' ');
            auto tones = split(parts.back(), ' ');
            // printf("line %d\n", pinyin_dict.size() + 1);
            // printf("parts[0]: %s\n", parts[0].c_str());
            // printf("parts[1]: %s\n", parts[1].c_str());
            // printf("parts[2]: %s\n", parts[2].c_str());

            pinyin_dict.insert({character, std::make_pair(pinyin_parts, tones)});
        }

        // if (pinyin_dict.size() >= 50)
        //     break;
    }
    return true;
}

// def get_pinyin_and_tones(sentence, pinyin_dict):
//     '''
//     输入句子和发音字典
//     返回对应的拼音列表和音调列表
//     '''
//     pinyin_list = []
//     tone_list = []
//     for char in sentence:
//         if char in pinyin_dict:
//             pinyin, tone = pinyin_dict[char]
//             pinyin_list.append(pinyin)
//             tone_list.append(tone)
//         else:
//             pinyin_list.append(char)
//             tone_list.append('')  # 若找不到对应拼音，音调留空

//     pinyin_list = [item for sublist in pinyin_list for item in sublist]
//     tone_list = [item for sublist in tone_list for item in sublist]
//     tone_list_ = []
//     for x in tone_list:
//         tones = x.split()
//         for xx in tones:
//             tone_list_.append(int(xx))
//     tone_list = tone_list_
//     return pinyin_list, tone_list
static void get_pinyin_and_tones(const std::string& sentence, const PINYIN_DICT_TYPE& pinyin_dict, std::vector<std::string>& pinyin_list, std::vector<int64_t>& tone_list) {
    auto utf8_strs = splitEachChar(sentence);
    for (auto s : utf8_strs) {
        // printf("s = %s\n", s.c_str());
        if (pinyin_dict.find(s) != pinyin_dict.end()) {
            for (auto p : pinyin_dict.at(s).first)
                pinyin_list.push_back(p);
            for (auto t : pinyin_dict.at(s).second)
                tone_list.push_back(t[0] - '0');
        } else {
            pinyin_list.push_back(s);
            tone_list.push_back(0L);
        }
    }
}

// def load_syllable_dict(file_path="../models/melo_tokens.txt"):
//     syllable_dict = {}
//     with open(file_path, 'r', encoding='utf-8') as file:
//         for line in file:
//             parts = line.strip().split()
//             if len(parts) == 2:
//                 syllable = parts[0]
//                 number = parts[1]
//                 syllable_dict[syllable] = int(number)
//     return syllable_dict
static bool load_syllable_dict(const std::string& token_file, std::unordered_map<std::string, int>& syllable_dict) {
    std::ifstream ifs(token_file);
    if (!ifs.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        auto parts = split(line, ' ');
        if (parts.size() == 2) {
            auto syllable = parts[0];
            auto number = parts[1];
            syllable_dict.insert({syllable, std::stoi(number)});
        }
    }

    return true;
}

// def replace_syllables_with_numbers(array, syllable_dict):
//     return [syllable_dict.get(item, item) for item in array]
static void replace_syllables_with_numbers(const std::vector<std::string>& pinyin_list, const std::unordered_map<std::string, int>& syllable_dict, std::vector<int64_t>& replaced_array) {
    for (const auto& pinyin : pinyin_list) {
        if (syllable_dict.find(pinyin) != syllable_dict.end()) {
            replaced_array.push_back(syllable_dict.at(pinyin));
        }
    }
}

// def insert_zeros(arr):
//     '''
//     输入 1 2 3
//     输出0 1 0 2 0 3 0
//     '''
//     # 计算插入0后新数组的长度
//     new_length = len(arr) * 2
//     # 创建新数组，初始值全部为0
//     new_arr = np.zeros(new_length, dtype=arr.dtype)
//     # 将原始数组的值按顺序插入到新数组中
//     new_arr[1::2] = arr
//     new_arr = np.append(new_arr,0)
//     return new_arr
static std::vector<int64_t> insert_zeros(const std::vector<int64_t>& arr) {
    size_t new_length = arr.size() * 2;
    std::vector<int64_t> new_arr(new_length, 0);
    for (size_t i = 1; i < new_arr.size(); i+=2) {
        new_arr[i] = arr[i / 2];
    }
    new_arr.push_back(0);
    return new_arr;
}

static int calc_product(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (auto i : dims)
        result *= i;
    return result;
}

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("encoder", 'e', "encoder onnx", true, "");
    cmd.add<std::string>("flow", 'f', "flow axmodel", true, "");
    cmd.add<std::string>("decoder", 'd', "decoder axmodel", true, "");
    cmd.add<std::string>("lexicon", 'l', "melo_lexicon_zh.txt", true, "");
    cmd.add<std::string>("token", 't', "melo_tokens.txt", true, "");
    cmd.add<std::string>("g", 0, "g.bin", true, "");

    cmd.add<std::string>("sentence", 's', "input sentence", true, "");
    cmd.add<std::string>("wav", 'w', "wav file", true, "");

    cmd.add<float>("speed", 0, "speak speed", false, 1.0f);
    cmd.add<int>("sample_rate", 0, "sample rate", false, 44100);
    cmd.parse_check(argc, argv);

    auto encoder_file   = cmd.get<std::string>("encoder");
    auto flow_file      = cmd.get<std::string>("flow");
    auto decoder_file   = cmd.get<std::string>("decoder");
    auto lexicon_file   = cmd.get<std::string>("lexicon");
    auto token_file     = cmd.get<std::string>("token");
    auto g_file         = cmd.get<std::string>("g");

    auto sentence       = cmd.get<std::string>("sentence");
    auto wav_file       = cmd.get<std::string>("wav");

    auto speed          = cmd.get<float>("speed");
    auto sample_rate    = cmd.get<int>("sample_rate");

    printf("encoder: %s\n", encoder_file.c_str());
    printf("flow: %s\n", flow_file.c_str());
    printf("decoder: %s\n", decoder_file.c_str());
    printf("lexicon: %s\n", lexicon_file.c_str());
    printf("token: %s\n", token_file.c_str());
    printf("sentence: %s\n", sentence.c_str());
    printf("wav: %s\n", wav_file.c_str());
    printf("speed: %f\n", speed);
    printf("sample_rate: %d\n", sample_rate);

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }

    // Load pinyin dict
    PINYIN_DICT_TYPE pinyin_dict;
    if (!load_pinyin_dict(pinyin_dict, lexicon_file)) {
        printf("Load pinyin dict failed!\n");
        return -1;
    }

    std::vector<std::string> pinyin_list;
    std::vector<int64_t> tone_list;
    get_pinyin_and_tones(sentence, pinyin_dict, pinyin_list, tone_list);

    std::unordered_map<std::string, int> syllable_dict;
    if (!load_syllable_dict(token_file, syllable_dict)) {
        printf("Load syllable dict failed!\n");
        return -1;
    }

    // 替换音节为对应数字
    std::vector<int64_t> replaced_array;
    replace_syllables_with_numbers(pinyin_list, syllable_dict, replaced_array);

    // replaced_array = np.pad(replaced_array, pad_width=1, mode='constant', constant_values=0)
    // tone_array = np.pad(tone_list, pad_width=1, mode='constant', constant_values=0)
    // langids = np.zeros_like(tone_array) + 3
    replaced_array.insert(replaced_array.begin(), 0L);
    replaced_array.push_back(0L);
    tone_list.insert(tone_list.begin(), 0L);
    tone_list.push_back(0L);

    // x_tst = insert_zeros(replaced_array).reshape((1,-1))
    // xlen = len(x_tst[0])
    // tones = insert_zeros(tone_array).reshape((1,-1))
    // langids = insert_zeros(langids).reshape((1,-1))
    auto x_tst = insert_zeros(replaced_array);
    int64_t xlen = x_tst.size();
    auto tones = insert_zeros(tone_list);

    // x_tst = np.array(x_tst,dtype=np.int64)
    // xlen = np.array([xlen],dtype=np.int64)
    // tones = np.array(tones,dtype=np.int64)

    // langids = np.zeros_like(x_tst) + 3
    // langids =np.array(langids,dtype=np.int64)
    // bert = np.zeros((1,1024,len(x_tst[0])),dtype=np.float32)
    // jabert = np.zeros((1,768,len(x_tst[0])),dtype=np.float32)
    std::vector<int64_t> langids(xlen, 3);
    std::vector<float> bert(1024 * xlen, 0);
    std::vector<float> jabert(768 * xlen, 0);

    // printf("tone len: %d\n", tones.size());

    // for (auto i : x_tst) {
    //     printf("%d ", i);
    // }
    // printf("\n");
    // for (auto i : tones) {
    //     printf("%d ", i);
    // }
    // printf("\n");
    // for (auto i : langids) {
    //     printf("%d ", i);
    // }
    // printf("\n");

    // Read g.bin
    std::vector<float> g(256, 0);
    FILE* fp = fopen(g_file.c_str(), "rb");
    if (!fp) {
        printf("Open %s failed!\n", g_file.c_str());
        return -1;
    }
    fread(g.data(), sizeof(float), g.size(), fp);
    fclose(fp);

    printf("Load encoder\n");
    OnnxWrapper encoder;
    if (0 != encoder.Init(encoder_file)) {
        printf("encoder init failed!\n");
        return -1;
    }

    float noise_scale   = 0.3f;
    float length_scale  = 1.0 / speed;
    float noise_scale_w = 0.6f;
    float sdp_ratio     = 0.2f;
    auto encoder_output = encoder.Run(x_tst, xlen, g, tones, langids, bert, jabert, noise_scale, length_scale, noise_scale_w, sdp_ratio);
    // auto encoder_output = encoder.RunTest();
    float* zp_data = encoder_output.at(0).GetTensorMutableData<float>();
    float* ymask_data = encoder_output.at(1).GetTensorMutableData<float>();
    auto zp_info = encoder_output.at(0).GetTensorTypeAndShapeInfo();
    auto ymask_info = encoder_output.at(1).GetTensorTypeAndShapeInfo();
    auto zp_shape = zp_info.GetShape();
    auto ymask_shape = ymask_info.GetShape();
    int zp_size = calc_product(zp_shape);
    int ymask_size = calc_product(ymask_shape);
    std::vector<float> zp(zp_data, zp_data + zp_size);
    std::vector<float> ymask(ymask_data, ymask_data + ymask_size);
    
    int mellen = ymask.size();
    printf("mellen: %d\n", mellen);
    int segsize = 120;
    int padsize = 10;
    std::vector<float> ymaskseg(segsize, 1);
    std::vector<float> wavlist;
    int i = 0;

    printf("Load flow model\n");
    EngineWrapper flow_model;
    if (0 != flow_model.Init(flow_file.c_str())) {
        printf("Init flow model failed!\n");
        return -1;
    }

    printf("Load decoder model\n");
    EngineWrapper decoder_model;
    if (0 != decoder_model.Init(decoder_file.c_str())) {
        printf("Init decoder model failed!\n");
        return -1;
    }

    int flowout_size = flow_model.GetOutputSize(0);
    std::vector<float> flowout(flowout_size / 4);

    int wav_size = decoder_model.GetOutputSize(0);
    std::vector<float> decoder_output(wav_size / 4);

    // while(i+segsize<=mellen):
    //     segz = zp[:,:,i:i+segsize]
    //     i += segsize-2*padsize
        
    //     segz = segz.flatten()

    //     x = sessf.run(input_feed={'z': segz, 'ymask': ymaskseg, 'g': g})
    //     flowout = x["6797"].flatten()

    //     x = sessd.run(input_feed={'z': flowout, 'g': g})

    //     wav = np.array(x["827"], dtype=np.float32).flatten()
    //     wav *= 5
    //     wavlist.append(wav[padsize*512:-padsize*512])
    int segz_i = 0;
    char segz_filename[32] = {0};
    while (i + segsize <= mellen) {
        std::vector<float> segz(192 * 120);
        for (int n = 0; n < 192; n++) {
            memcpy(segz.data() + n * 120, zp.data() + n * mellen + i, sizeof(float) * 120);
        }

        flow_model.SetInput(segz.data(), 0);
        flow_model.SetInput(ymaskseg.data(), 1);
        flow_model.SetInput(g.data(), 2);
        if (0 != flow_model.RunSync()) {
            printf("Run flow model failed!\n");
            return -1;
        }
        flow_model.GetOutput(flowout.data(), 0);

        i += segsize - 2 * padsize;

        decoder_model.SetInput(flowout.data(), 0);
        decoder_model.SetInput(g.data(), 1);
        if (0 != decoder_model.RunSync()) {
            printf("Run decoder model failed!\n");
            return -1;
        }
        decoder_model.GetOutput(decoder_output.data(), 0);
        
        for (size_t n = padsize * 512; n < decoder_output.size() - padsize * 512; n++) {
            wavlist.push_back(decoder_output[n] * 5);
        }
    }

    printf("wav len: %d\n", wavlist.size());
    AudioFile<float> audio_file;
    std::vector<std::vector<float> > audio_samples{wavlist};
    audio_file.setAudioBuffer(audio_samples);
    audio_file.setSampleRate(sample_rate);
    if (!audio_file.save(wav_file)) {
        printf("Save audio file failed!\n");
        return -1;
    }

    printf("Saved audio to %s\n", wav_file.c_str());

    return 0;
}