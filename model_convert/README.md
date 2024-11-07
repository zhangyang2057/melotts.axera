# 模型转换

## 创建虚拟环境   
```conda create -n melotts python=3.9```  
```conda activate melotts```  

## 安装依赖
```cd model_convert```  
```pip install requirements.txt```  

## 转换模型(PyTorch->ONNX)
由于需要到huggingface下载PyTorch权重，可能需要设置代理，可手动修改convert.py的PROXIES参数  
```python convert.py```  

## 转换模型(ONNX->Axera)
```pulsar2 build --input decoder.onnx --config config_decoder_u16.json --output_dir decoder --output_name decoder.axmodel --target_hardware AX650 --npu_mode NPU3 --compiler.check 0```  

转换完成后会生成decoder/decoder.axmodel