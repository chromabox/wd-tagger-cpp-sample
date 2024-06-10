# wd-tagger-cpp-sample
これは`wd-tagger`(waifu diffusion tagger)をc++で使う場合のサンプルです  
pythonのサンプルは色々あるのですが、c++で使うサンプルが見当たらない感じなので作りました  

参考：https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2


## ビルド方法(ubuntu/linux):

cmake とか opencv周りはインストールしておきます
```
$ sudo apt install build-essential cmake libopencv-dev
```

その後cloneしてから  
```bash
$ ./build.sh
```
でとりあえずビルド出来ます。  
onnxruntimeはビルド時にonnxlibにダウンロードしてくれます。
また、wdのtaggerモデルもビルド時にmodelsにダウンロードしてくれます。

## TODO:
- Windowsでの例を入れる