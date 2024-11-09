# 模型转换

## 创建虚拟环境

```
conda create -n melotts python=3.9  
conda activate melotts
```

## 安装依赖

```
git clone https://github.com/ml-inory/melotts.axera.git
cd model_convert 
sudo apt install libmecab-dev
pip install unidic-lite fugashi
pip install -r requirements.txt
```  

## 转换模型(PyTorch -> ONNX)

```
python convert.py
```  

## 转换模型(ONNX -> Axera)

```
pulsar2 build --input decoder.onnx --config config_decoder_u16.json --output_dir decoder --output_name decoder.axmodel --target_hardware AX650 --npu_mode NPU3 --compiler.check 0
```  

转换完成后会生成 decoder/decoder.axmodel
