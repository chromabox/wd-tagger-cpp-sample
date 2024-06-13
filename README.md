# wd-tagger-cpp-sample
  
これは`wd-tagger`(waifu diffusion tagger)をc++とonnxruntimeで使う場合のサンプルです  
コマンドラインで動作します  
  
`wd-tagger`とは簡単に説明すると指定された画像をAIで分析して、その要素を示す文字（タグ）を出力するというものです  
単純に画像の要素を調べるのにも使用されますし、追加学習(LoRA)用のキャプションの自動生成にも使用されます  
  
これらのpythonで書かれたサンプルは色々あるのですが、c++で使うサンプルが見当たらない感じなので作りました  

参考：  
https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2
https://huggingface.co/spaces/SmilingWolf/wd-tagger


## ビルド方法(ubuntu/linux):

はじめにビルドに必要な`cmake`とか`opencv`周りはインストールしておきます
`onnxruntime`や`wd-tagger`のモデルのダウンロードに`wget`を使用していますのでそれも入れておきます
```
$ sudo apt install build-essential cmake libopencv-dev wget
```

その後cloneしてから  
```bash
$ ./build.sh
```
でとりあえずビルド出来ます。  
onnxruntimeはビルド時に自動的にonnxlibにダウンロードしています。
また、wdのtaggerモデルもビルド時にmodelsにダウンロードしています。

## 使い方

buildディレクトリにwdtagger実行ファイルがあるので、以下のようにして実行します
```
$ cd build
$ ./wdtagger 画像ファイル名
```
指定する画像ファイルのチャンネル数は3でないといけません  
つまり、透過pngやアルファチャンネルが含まれている画像ファイルはエラーになります  

例として、私のアバター(https://avatars.githubusercontent.com/u/846461?v=4)を読ませると次のような結果になります  
  
タグの横に記されている値が高いほど、その要素が含まれている可能性が高いことを示します  
`tags`と`charas`はトップ10まで表示されています  
サンプルでは結果値の少なさに関わらずトップ10すべて表示しますが、`tags`は0.35未満、`charas`は0.85未満のタグはあまり信用できないかもしれない結果ということのようです
  
```
$ ./wdtagger 846461.png 
read label file...
read label ok.
loading wd-tagger model... 
load wd-tagger model success
Running predict...
predict success.
result -----  
ratings: 
   general: 0.0733535
   sensitive: 0.920688
   questionable: 0.00349146
   explicit: 0.000387698
tags: 
   1girl: 0.991317
   solo: 0.98317
   instrument: 0.866745
   guitar: 0.774588
   brown_hair: 0.746188
   purple_eyes: 0.68452
   animal_ears: 0.629489
   short_hair: 0.615854
   bow: 0.60101
   chibi: 0.375334
charas: 
   ushiromiya_maria: 0.667614
   ryuuguu_rena: 0.0163996
   chen: 0.00753877
   hirasawa_yui: 0.00524783
   yakumo_ran: 0.00188139
   houjou_satoko: 0.00141031
   misty_(pokemon): 0.00120723
   souryuu_asuka_langley: 0.0010246
   may_(pokemon): 0.00101718
   alice_margatroid: 0.000895023
```


## モデルの変え方

モデルは`wd-v1-4-vit-tagger-v2`を使用しています  
これはあくまでサンプルなので、コードにベタに書いています  
  
`wd-tagger`には他にもモデルがあり、以下の手順で変更することが出来るかと思います  
  
1.modelsディレクトリにダウンロードしたモデルを置く
形式は次のようにします  
  
models/モデル名/model.onnx  
models/モデル名/selected_tags.csv  
  
2.src/wdtagger.cppのWD_MODEL_ONNXとWD_LABEL_FNAMEを書き換えます
```
#define WD_MODEL_ONNX	"../models/wd-vit-tagger-v2/model.onnx"
#define WD_LABEL_FNAME	"../models/wd-vit-tagger-v2/selected_tags.csv"
```
3.リビルドします


## TODO:
- Windowsでの例を入れる