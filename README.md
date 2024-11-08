# melotts.axera
MeloTTS demo on Axera AX650

目前模型分成了encoder、decoder两部分，encoder部分尚未转成axmodel（目前通过onnxruntime运行）。  
models/下的模型为中英混合模型，如需自行转换请参考[模型转换](/model_convert/README.md)

TBD：
- [ ] encoder转成axmodel
- [ ] 效果与官方repo对齐

## 中文输入(板上运行)
```locale-gen C.utf8```  
```update-locale LANG=C.utf8```

重新打开终端即可进行中文输入

## Python(板上运行)
### Requirements
```apt-get install libsndfile1-dev```  
```mkdir /opt/site-packages```  
```pip3 install -r requirements.txt --prefix=/opt/site-packages```  

将这两行放到/root/.bashrc  
(实际添加的路径需要自行检查)
```export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages```  
```export PATH=$PATH:/opt/site-packages/local/bin```  
 重新连接终端

### 运行
```cd python```  
```python3 melotts.py -s 要生成语音的句子```  

### 示例
```python3 melotts.py -s 爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用```  
输出音频:
[demo.wav](https://raw.githubusercontent.com/ml-inory/melotts.axera/main/demo.wav)
## Cpp(交叉编译)
下载BSP
```
bash download_bsp.sh
```
编译
```
bash build.sh
```
板上运行
```
./install/bin/melotts -l ../models/melo_lexicon_zh.txt -t ../models/melo_tokens.txt -e ../models/enc-sim.onnx -f ../models/flow.axmodel -d ../models/decoder.axmodel --g ../models/g.bin -w test_cn.wav -s 爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾 驶、机器人的海量普惠的应用
```