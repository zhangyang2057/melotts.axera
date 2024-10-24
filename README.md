# melotts.axera
MeloTTS demo on Axera AX650

目前模型分成了encoder、flow、decoder三部分，encoder部分尚未转成axmodel（目前通过onnxruntime运行）。

TBD：
- [ ] encoder转成axmodel
- [ ] 效果与官方repo对齐

## For Chinese input(Run on board)
```locale-gen C.utf8```  
```update-locale LANG=C.utf8```  

## Python(Run on board)
### Requirements
```apt-get install libsndfile1-dev```  
```mkdir /opt/site-packages```  
```pip3 install -r requirements.txt --prefix=/opt/site-packages```  

Put  
```export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages```  
```export PATH=$PATH:/opt/site-packages/local/bin```  
to /root/.bashrc  

### Run
```python3 onnx_infer_stream.py --sentence 要生成语音的句子```  

## Cpp(Cross compile)
Download BSP
```
bash download_bsp.sh
```
Build
```
bash build.sh
```
Run on board
```
./build/install/bin/melotts -l ../models/melo_lexicon_zh.txt -t ../models/melo_tokens.txt -e ../models/enc-sim.onnx -f ../models/flow.axmodel -d ../models/decoder.axmodel --g ../models/g.bin -w test_cn.wav -s 爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾 驶、机器人的海量普惠的应用
```