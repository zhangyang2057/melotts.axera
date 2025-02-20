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
#include <cmath>
#include <ctime>
#include <chrono>
#include "cmdline.hpp"
#include "AudioFile.h"
#include "Lexicon.hpp"
#if defined(ONNX)
#include "OnnxWrapper.hpp"
#else
#include "NncaseWrapper.hpp"
#endif()

class ScopedTiming {
public:
    ScopedTiming(std::string info = "ScopedTiming") : m_info(info) {
        m_start = std::chrono::steady_clock::now();
    }

    ~ScopedTiming() {
        m_stop = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double,std::milli>(m_stop - m_start).count();
        std::cout << m_info << " took " << elapsed_ms << " ms" << std::endl;
    }

private:
    std::string m_info;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_stop;
};

template <typename T>
void read_binary_file(const std::string &file_name, std::vector<T> &v)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    v.resize(len / sizeof(T));
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(v.data()), len);
    ifs.close();
}

void write_binary_file(const char *file_name, const char *buf, size_t size)
{
    std::ofstream ofs(file_name, std::ios::out | std::ios::binary);
    ofs.write(buf, size);
    ofs.close();
}

static std::vector<int> intersperse(const std::vector<int>& lst, int item) {
    std::vector<int> result(lst.size() * 2 + 1, item);
    for (size_t i = 1; i < result.size(); i+=2) {
        result[i] = lst[i / 2];
    }
    return result;
}

static int calc_product(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (auto i : dims)
        result *= i;
    return result;
}

int main(int argc, char** argv) {
    cmdline::parser cmd;
#if defined(ONNX)
    cmd.add<std::string>("encoder", 'e', "encoder onnx", false, "../models/encoder-zh.onnx");
    cmd.add<std::string>("decoder", 'd', "decoder onnx", false, "../models/decoder-zh.onnx");
#else
    cmd.add<std::string>("encoder", 'e', "encoder kmodel", false, "../models/encoder-zh.kmodel");
    cmd.add<std::string>("decoder", 'd', "decoder kmodel", false, "../models/decoder-zh.kmodel");
#endif
    cmd.add<std::string>("lexicon", 'l', "lexicon.txt", false, "../models/lexicon.txt");
    cmd.add<std::string>("token", 't', "tokens.txt", false, "../models/tokens.txt");
    cmd.add<std::string>("g", 0, "g.bin", false, "../models/g-zh_mix_en.bin");
    cmd.add<std::string>("sentence", 's', "input sentence", false, "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用");

    cmd.add<std::string>("wav", 'w', "wav file", false, "output.wav");

    cmd.add<float>("speed", 0, "speak speed", false, 0.8f);
    cmd.add<int>("sample_rate", 0, "sample rate", false, 44100);
    cmd.parse_check(argc, argv);

    auto encoder_file   = cmd.get<std::string>("encoder");
    auto decoder_file   = cmd.get<std::string>("decoder");
    auto lexicon_file   = cmd.get<std::string>("lexicon");
    auto token_file     = cmd.get<std::string>("token");
    auto g_file         = cmd.get<std::string>("g");

    auto sentence       = cmd.get<std::string>("sentence");
    auto wav_file       = cmd.get<std::string>("wav");

    auto speed          = cmd.get<float>("speed");
    auto sample_rate    = cmd.get<int>("sample_rate");

    printf("encoder: %s\n", encoder_file.c_str());
    printf("decoder: %s\n", decoder_file.c_str());
    printf("lexicon: %s\n", lexicon_file.c_str());
    printf("token: %s\n", token_file.c_str());
    printf("sentence: %s\n", sentence.c_str());
    printf("wav: %s\n", wav_file.c_str());
    printf("speed: %f\n", speed);
    printf("sample_rate: %d\n", sample_rate);

    // Load lexicon
    Lexicon lexicon(lexicon_file, token_file);

    // Convert sentence to phones and tones
    std::vector<int> phones_bef, tones_bef;
    lexicon.convert(sentence, phones_bef, tones_bef);

    // for (auto p : phones_bef) {
    //     printf("%d ", p);
    // }
    // printf("\n");

    // Add blank between words
    auto phones = intersperse(phones_bef, 0);
    auto tones = intersperse(tones_bef, 0);
    int phone_len = phones.size();

    std::vector<int> langids(phone_len, 3);

    // Read g.bin
    std::vector<float> g(256, 0);
    FILE* fp = fopen(g_file.c_str(), "rb");
    if (!fp) {
        printf("Open %s failed!\n", g_file.c_str());
        return -1;
    }
    fread(g.data(), sizeof(float), g.size(), fp);
    fclose(fp);

#if defined(ONNX)
    OnnxWrapper encoder;
    int ret = 0;
    {
        ScopedTiming st("load encoder");
        ret = encoder.Init(encoder_file);
    }

    if (0 != ret)
    {
        printf("encoder init failed!\n");
        return -1;
    }

    OnnxWrapper decoder;
    {
        ScopedTiming st("load decoder");
        ret = decoder.Init(decoder_file);
    }

    if (0 != ret) {
        printf("decoder init failed!\n");
        return -1;
    }
#else
    NncaseModel encoder(encoder_file);
    NncaseModel decoder(decoder_file);
#endif

    float noise_scale   = 0.3f;
    float length_scale  = 1.0 / speed;
    float noise_scale_w = 0.6f;
    float sdp_ratio     = 0.2f;

#if defined(ONNX)
    write_binary_file("phone.bin", reinterpret_cast<char*>(phones.data()), phones.size() * sizeof(int));
    write_binary_file("tone.bin", reinterpret_cast<char*>(tones.data()), tones.size() * sizeof(int));
    write_binary_file("language.bin", reinterpret_cast<char *>(langids.data()), langids.size() * sizeof(int));
    write_binary_file("g.bin", reinterpret_cast<char *>(g.data()), g.size() * sizeof(float));
    write_binary_file("noise_scale.bin", reinterpret_cast<char *>(&noise_scale), sizeof(noise_scale));
    write_binary_file("noise_scale_w.bin", reinterpret_cast<char *>(&noise_scale_w), sizeof(noise_scale_w));
    write_binary_file("length_scale.bin", reinterpret_cast<char *>(&length_scale), sizeof(length_scale));
    write_binary_file("sdp_ratio.bin", reinterpret_cast<char *>(&sdp_ratio), sizeof(sdp_ratio));

    std::vector<Ort::Value> encoder_output;
    {
        ScopedTiming st("encoder run");
        encoder_output = encoder.Run(phones, tones, langids, g, noise_scale, noise_scale_w, length_scale, sdp_ratio);
    }

    float* zp_data = encoder_output.at(0).GetTensorMutableData<float>();
    int audio_len = encoder_output.at(2).GetTensorMutableData<int>()[0];
    auto zp_info = encoder_output.at(0).GetTensorTypeAndShapeInfo();
    auto zp_shape = zp_info.GetShape();
    write_binary_file("zp.bin", reinterpret_cast<char*>(zp_data), zp_shape[0] * zp_shape[1] * zp_shape[2] * sizeof(float));
#else
    std::vector<nncase::value_t> inputs;
    auto entry = encoder.entry();

    // input 0: phone
    auto type = entry->parameter_type(0).expect("parameter type out of index");
    auto ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    auto data_type = ts_type->dtype()->typecode();
    nncase::dims_t phone_shape{phones.size()};
    auto phone_tensor = nncase::runtime::host_runtime_tensor::create(data_type, phone_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto phone_buffer = phone_tensor->buffer().as_host().unwrap_or_throw();
    auto phone_mapped = phone_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto phone_ptr = phone_mapped.buffer().as_span<int>().data();
    std::memcpy(phone_ptr, phones.data(), phones.size() * sizeof(int));
    phone_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(phone_tensor);

    // input 1: tone
    type = entry->parameter_type(1).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t tone_shape{tones.size()};
    auto tone_tensor = nncase::runtime::host_runtime_tensor::create(data_type, tone_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto tone_buffer = tone_tensor->buffer().as_host().unwrap_or_throw();
    auto tone_mapped = tone_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto tone_ptr = tone_mapped.buffer().as_span<int>().data();
    std::memcpy(tone_ptr, tones.data(), tones.size() * sizeof(int));
    tone_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(tone_tensor);

    // input 2: language
    type = entry->parameter_type(2).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t language_shape{langids.size()};
    auto language_tensor = nncase::runtime::host_runtime_tensor::create(data_type, language_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto language_buffer = language_tensor->buffer().as_host().unwrap_or_throw();
    auto language_mapped = language_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto language_ptr = language_mapped.buffer().as_span<int>().data();
    std::memcpy(language_ptr, langids.data(), langids.size() * sizeof(int));
    language_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(language_tensor);

    // input 3: g
    type = entry->parameter_type(3).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t g_shape{1, g.size(), 1};
    auto g_tensor = nncase::runtime::host_runtime_tensor::create(data_type, g_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto g_buffer = g_tensor->buffer().as_host().unwrap_or_throw();
    auto g_mapped = g_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto g_ptr = g_mapped.buffer().as_span<float>().data();
    std::memcpy(g_ptr, g.data(), g.size() * sizeof(float));
    g_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(g_tensor);

    // input 4: noise_scale
    type = entry->parameter_type(4).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t noise_scale_shape{1};
    auto noise_scale_tensor = nncase::runtime::host_runtime_tensor::create(data_type, noise_scale_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto noise_scale_buffer = noise_scale_tensor->buffer().as_host().unwrap_or_throw();
    auto noise_scale_mapped = noise_scale_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto noise_scale_ptr = noise_scale_mapped.buffer().as_span<float>().data();
    noise_scale_ptr[0] = noise_scale;
    noise_scale_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(noise_scale_tensor);

    // input 5: noise_scale_w
    type = entry->parameter_type(5).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t noise_scale_w_shape{1};
    auto noise_scale_w_tensor = nncase::runtime::host_runtime_tensor::create(data_type, noise_scale_w_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto noise_scale_w_buffer = noise_scale_w_tensor->buffer().as_host().unwrap_or_throw();
    auto noise_scale_w_mapped = noise_scale_w_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto noise_scale_w_ptr = noise_scale_w_mapped.buffer().as_span<float>().data();
    noise_scale_w_ptr[0] = noise_scale_w;
    noise_scale_w_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(noise_scale_w_tensor);

    // input 6: length_scale
    type = entry->parameter_type(6).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t length_scale_shape{1};
    auto length_scale_tensor = nncase::runtime::host_runtime_tensor::create(data_type, length_scale_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto length_scale_buffer = length_scale_tensor->buffer().as_host().unwrap_or_throw();
    auto length_scale_mapped = length_scale_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto length_scale_ptr = length_scale_mapped.buffer().as_span<float>().data();
    length_scale_ptr[0] = length_scale;
    length_scale_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(length_scale_tensor);

    // input 7: sdp_ratio
    type = entry->parameter_type(7).expect("parameter type out of index");
    ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
    data_type = ts_type->dtype()->typecode();
    nncase::dims_t sdp_ratio_shape{1};
    auto sdp_ratio_tensor = nncase::runtime::host_runtime_tensor::create(data_type, sdp_ratio_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
    auto sdp_ratio_buffer = sdp_ratio_tensor->buffer().as_host().unwrap_or_throw();
    auto sdp_ratio_mapped = sdp_ratio_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
    auto sdp_ratio_ptr = sdp_ratio_mapped.buffer().as_span<float>().data();
    sdp_ratio_ptr[0] = sdp_ratio;
    sdp_ratio_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
    inputs.push_back(sdp_ratio_tensor);

#if 1
    // run
    nncase::value_t outs;
    {
        ScopedTiming st("encoder run");
        outs = encoder.run(inputs);
    }
    auto outputs = outs.as<nncase::tuple>().unwrap();

    // get output 0
    auto z_p_tensor = outputs->fields()[0].as<nncase::tensor>().unwrap_or_throw();
    auto z_p_buffer = z_p_tensor->buffer().as_host().unwrap_or_throw();
    auto z_p_mapped = z_p_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    float *zp_data = z_p_mapped.buffer().as_span<float>().data();
    auto zp_shape = z_p_tensor->shape();

    // get output 2
    auto audio_len_tensor = outputs->fields()[2].as<nncase::tensor>().unwrap_or_throw();
    auto audio_len_buffer = audio_len_tensor->buffer().as_host().unwrap_or_throw();
    auto audio_len_mapped = audio_len_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
    auto audio_len_data = audio_len_mapped.buffer().as_span<int>();
    int audio_len = audio_len_data[0];

#else
    std::vector<float> v_zp;
    read_binary_file("zp.bin", v_zp);
    float *zp_data = v_zp.data();
    std::vector<int64_t> zp_shape = {1, 192, 1034};
    int audio_len = 529408;
#endif
#endif

    std::cout << "audio_len = " << audio_len << std::endl;

    std::cout << "zp_shape = ";
    for (size_t i = 0; i < zp_shape.size(); i++)
    {
        std::cout << zp_shape[i] << " ";
    }
    std::cout << std::endl;

    int zp_size = 1 * 192 * 128;
    int dec_len = zp_size / zp_shape[1];
    int audio_slice_len = 65536;
    std::vector<float> decoder_output(audio_slice_len);

    int dec_slice_num = int(std::ceil(zp_shape[2] * 1.0 / dec_len));
    std::cout << "dec_slice_num = " << dec_slice_num << std::endl;
    std::vector<float> wavlist;
    for (int i = 0; i < dec_slice_num; i++) {
        std::vector<float> zp(zp_size, 0);
        int actual_size = (i + 1) * dec_len < zp_shape[2] ? dec_len : zp_shape[2] - i * dec_len;
        for (int n = 0; n < zp_shape[1]; n++) {
            memcpy(zp.data() + n * dec_len, zp_data + n * zp_shape[2] + i * dec_len, sizeof(float) * actual_size);
        }

#if defined(ONNX)
        std::vector<Ort::Value> dec_out;
        {
            ScopedTiming st("decoder run");
            dec_out = decoder.Run(zp, g);
        }
        float* audio = dec_out.at(0).GetTensorMutableData<float>();
#else
        inputs.clear();
        entry = decoder.entry();

        // input 0: phone
        type = entry->parameter_type(0).expect("parameter type out of index");
        ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
        data_type = ts_type->dtype()->typecode();
        nncase::dims_t z_p_shape{1, 192, 128};
        auto z_p_tensor = nncase::runtime::host_runtime_tensor::create(data_type, z_p_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
        auto z_p_buffer = z_p_tensor->buffer().as_host().unwrap_or_throw();
        auto z_p_mapped = z_p_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
        auto z_p_ptr = z_p_mapped.buffer().as_span<float>().data();
        memcpy((void *)z_p_ptr, (void *)zp.data(), zp.size() * sizeof(float));
        z_p_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
        inputs.push_back(z_p_tensor);


        // input 1: g
        type = entry->parameter_type(1).expect("parameter type out of index");
        ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
        data_type = ts_type->dtype()->typecode();
        nncase::dims_t g_shape{1, 256, 1};
        auto g_tensor = nncase::runtime::host_runtime_tensor::create(data_type, g_shape, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
        auto g_buffer = g_tensor->buffer().as_host().unwrap_or_throw();
        auto g_mapped = g_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
        auto g_ptr = g_mapped.buffer().as_span<float>().data();
        std::memcpy((void *)g_ptr, (void *)g.data(), g.size() * sizeof(float));
        g_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
        inputs.push_back(g_tensor);

        // run
        nncase::value_t out;
        {
            ScopedTiming st("decoder run");
            out = decoder.run(inputs);
        }

        // get output 0
        auto audio_tensor = out.as<nncase::tensor>().unwrap_or_throw();
        auto audio_buffer = audio_tensor->buffer().as_host().unwrap_or_throw();
        auto audio_mapped = audio_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
        float *audio = audio_mapped.buffer().as_span<float>().data();
#endif
        memcpy((void *)decoder_output.data(), (void *)audio, audio_slice_len * sizeof(float));

        actual_size = (i + 1) * audio_slice_len < audio_len ? audio_slice_len : audio_len - i * audio_slice_len;
        std::cout << "i = " << i << ", actual_size = " << actual_size << std::endl;
        wavlist.insert(wavlist.end(), decoder_output.begin(), decoder_output.begin() + actual_size);
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
