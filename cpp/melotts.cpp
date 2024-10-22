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
#include "onnxruntime_cxx_api.h"

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
static void get_pinyin_and_tones(const std::string& sentence, const PINYIN_DICT_TYPE& pinyin_dict, std::vector<std::string>& pinyin_list, std::vector<int>& tone_list) {
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
            tone_list.push_back(0);
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


// def replace_syllables_with_numbers(array, syllable_dict):
//     return [syllable_dict.get(item, item) for item in array]


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

int main(int argc, char** argv) {
    cmdline::parser cmd;
    // cmd.add<std::string>("encoder", 'e', "encoder onnx", true, "");
    // cmd.add<std::string>("flow", 'f', "flow axmodel", true, "");
    // cmd.add<std::string>("decoder", 'd', "decoder axmodel", true, "");
    cmd.add<std::string>("lexicon", 'l', "melo_lexicon_zh.txt", true, "");
    cmd.add<std::string>("token", 't', "melo_tokens.txt", true, "");

    cmd.add<std::string>("sentence", 's', "input sentence", true, "");
    // cmd.add<std::string>("wav", 'w', "wav file", true, "");
    cmd.parse_check(argc, argv);

    // auto encoder_file   = cmd.get<std::string>("encoder");
    // auto flow_file      = cmd.get<std::string>("flow");
    // auto decoder_file   = cmd.get<std::string>("decoder");
    auto lexicon_file   = cmd.get<std::string>("lexicon");
    auto token_file     = cmd.get<std::string>("token");
    auto sentence       = cmd.get<std::string>("sentence");
    // auto wav_file       = cmd.get<std::string>("wav");

    // printf("encoder: %s\n", encoder_file.c_str());
    // printf("flow: %s\n", flow_file.c_str());
    // printf("decoder: %s\n", decoder_file.c_str());
    printf("lexicon: %s\n", lexicon_file.c_str());
    printf("token: %s\n", token_file.c_str());
    printf("sentence: %s\n", sentence.c_str());
    // printf("wav: %s\n", wav_file.c_str());

    // Load pinyin dict
    PINYIN_DICT_TYPE pinyin_dict;
    if (!load_pinyin_dict(pinyin_dict, lexicon_file)) {
        printf("Load pinyin dict failed!\n");
        return -1;
    }

    std::vector<std::string> pinyin_list;
    std::vector<int> tone_list;
    get_pinyin_and_tones(sentence, pinyin_dict, pinyin_list, tone_list);

    // printf("pinyin_list:\n");
    // for (auto c : pinyin_list) {
    //     printf("%s ", c.c_str());
    // }
    // printf("\n");

    // printf("tone_list:\n");
    // for (auto c : tone_list) {
    //     printf("%d ", c);
    // }
    // printf("\n");

    printf("Load encoder\n");
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "melotts-encoder");



    return 0;
}