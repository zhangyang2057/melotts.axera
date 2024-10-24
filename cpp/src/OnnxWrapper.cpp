#include "OnnxWrapper.hpp"

#include <string.h>
#include <stdlib.h>

int OnnxWrapper::Init(const std::string& model_file) {
    // set ort env
    m_ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, model_file.c_str());
    // 0. session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // GPU compatiable.
    // OrtCUDAProviderOptions provider_options;
    // session_options.AppendExecutionProvider_CUDA(provider_options);
    // #ifdef USE_CUDA
    //  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
    // #endif
 
    // 1. session
    m_session = new Ort::Session(m_ort_env, model_file.c_str(), session_options);
    // memory allocation and options
    Ort::AllocatorWithDefaultOptions allocator;
    // 2. input name & input dims
    m_input_num = m_session->GetInputCount();
    // for (int i = 0; i < m_input_num; i++) {
    //     std::string input_name(m_session->GetInputNameAllocated(i, allocator).get());
    //     printf("name[%d]: %s\n", i, input_name.c_str());
    // }

    // 4. output names & output dims
    m_output_num = m_session->GetOutputCount();
    // for (int i = 0; i < m_output_num; i++) {
    //     std::string output_name(m_session->GetOutputNameAllocated(i, allocator).get());
    //     printf("name[%d]: %s\n", i, output_name.c_str());
    // }
 
    return 0;
}

std::vector<Ort::Value> OnnxWrapper::Run(std::vector<int64_t>& x_tst, 
                                int64_t x_tst_l, 
                                std::vector<float>& g,
                                std::vector<int64_t>& tones,
                                std::vector<int64_t>& langids,
                                std::vector<float>& bert,
                                std::vector<float>& jabert,
                                float noise_scale,
                                float length_scale,
                                float noise_scale_w,
                                float sdp_ratio) {
    int64_t phonelen = x_tst.size();
    int64_t toneslen = tones.size();
    int64_t langidslen = langids.size();
    int64_t bertlen = bert.size();
    int64_t jabertlen = jabert.size();
     
    std::array<int64_t, 2> x_tst_dims{1, phonelen};
    std::array<int64_t, 1> x_tst_l_dims{1};
    std::array<int64_t, 3> g_dims{1, 256, 1};
    std::array<int64_t, 2> tones_dims{1, toneslen};
    std::array<int64_t, 2> langids_dims{1, langidslen};
    std::array<int64_t, 3> bert_dims{1, 1024, phonelen};
    std::array<int64_t, 3> jabert_dims{1, 768, phonelen};
    std::array<int64_t, 1> noise_scale_dims{1};
    std::array<int64_t, 1> length_scale_dims{1};
    std::array<int64_t, 1> noise_scale_w_dims{1};
    std::array<int64_t, 1> sdp_scale_dims{1};

    const char* input_names[] = {"x_tst", "x_tst_l", "g", "tones", "langids", "bert", "jabert", "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio"};
    const char* output_names[] = {"7251", "onnx::Unsqueeze_7168"};

    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_vals;
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, x_tst.data(), x_tst.size(), x_tst_dims.data(), x_tst_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, &x_tst_l, 1, x_tst_l_dims.data(), x_tst_l_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, g.data(), g.size(), g_dims.data(), g_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, tones.data(), tones.size(), tones_dims.data(), tones_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, langids.data(), langids.size(), langids_dims.data(), langids_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, bert.data(), bert.size(), bert_dims.data(), bert_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, jabert.data(), jabert.size(), jabert_dims.data(), jabert_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale, 1, noise_scale_dims.data(), noise_scale_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &length_scale, 1, length_scale_dims.data(), length_scale_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale_w, 1, noise_scale_w_dims.data(), noise_scale_w_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &sdp_ratio, 1, sdp_scale_dims.data(), sdp_scale_dims.size()));

    return m_session->Run(Ort::RunOptions{nullptr}, input_names, input_vals.data(), input_vals.size(), output_names, m_output_num);
}

std::vector<Ort::Value> OnnxWrapper::RunTest() {
    int xlen = 241;
    std::vector<int64_t> x_tst(xlen);
    int64_t x_tst_l = xlen;
    std::vector<float> g(256);
    std::vector<int64_t> tones(xlen);
    std::vector<int64_t> langids(xlen);
    std::vector<float> bert(1024 * xlen);
    std::vector<float> jabert(768 * xlen);
    float noise_scale   = 0.3f;
    float length_scale  = 1.0;
    float noise_scale_w = 0.6f;
    float sdp_ratio     = 0.2f;

    int64_t phonelen = x_tst.size();
    int64_t toneslen = tones.size();
    int64_t langidslen = langids.size();
    int64_t bertlen = bert.size();
    int64_t jabertlen = jabert.size();
     
    std::array<int64_t, 2> x_tst_dims{1, phonelen};
    std::array<int64_t, 1> x_tst_l_dims{1};
    std::array<int64_t, 3> g_dims{1, 256, 1};
    std::array<int64_t, 2> tones_dims{1, toneslen};
    std::array<int64_t, 2> langids_dims{1, langidslen};
    std::array<int64_t, 3> bert_dims{1, 1024, phonelen};
    std::array<int64_t, 3> jabert_dims{1, 768, phonelen};
    std::array<int64_t, 1> noise_scale_dims{1};
    std::array<int64_t, 1> length_scale_dims{1};
    std::array<int64_t, 1> noise_scale_w_dims{1};
    std::array<int64_t, 1> sdp_scale_dims{1};

    printf("phonelen: %d\n", phonelen);
    printf("toneslen: %d\n", toneslen);
    printf("langidslen: %d\n", langidslen);

    const char* input_names[] = {"x_tst", "x_tst_l", "g", "tones", "langids", "bert", "jabert", "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio"};
    const char* output_names[] = {"7251", "onnx::Unsqueeze_7168"};

    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_vals;
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, x_tst.data(), x_tst.size(), x_tst_dims.data(), x_tst_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, &x_tst_l, 1, x_tst_l_dims.data(), x_tst_l_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, g.data(), g.size(), g_dims.data(), g_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, tones.data(), tones.size(), tones_dims.data(), tones_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_handler, langids.data(), langids.size(), langids_dims.data(), langids_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, bert.data(), bert.size(), bert_dims.data(), bert_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, jabert.data(), jabert.size(), jabert_dims.data(), jabert_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale, 1, noise_scale_dims.data(), noise_scale_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &length_scale, 1, length_scale_dims.data(), length_scale_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale_w, 1, noise_scale_w_dims.data(), noise_scale_w_dims.size()));
    input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &sdp_ratio, 1, sdp_scale_dims.data(), sdp_scale_dims.size()));

    return m_session->Run(Ort::RunOptions{nullptr}, input_names, input_vals.data(), input_vals.size(), output_names, m_output_num);
}