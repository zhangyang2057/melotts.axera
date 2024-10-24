# melotts.axera
MeloTTS demo on Axera

## Requirements
```apt-get install libsndfile1-dev```  
```mkdir /opt/site-packages```  
```pip3 install -r requirements.txt --prefix=/opt/site-packages```

Put  
```export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages```
```export PATH=$PATH:/opt/site-packages/local/bin```  
to /root/.bashrc  

```locale-gen C.utf8```  
```update-locale LANG=C.utf8```  

## Run(Python)
```python3 onnx_infer_stream.py --sentence 要生成语音的句子```  