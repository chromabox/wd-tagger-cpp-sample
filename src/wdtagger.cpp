// The MIT License (MIT)
//
// Copyright (c) <2024> chromabox <chromarockjp@gmail.com>
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// wd-tagger (WaifuDiffusion Tagger)をc++で扱うサンプル

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include "onnx/onnxruntime_cxx_api.h"

#if defined _MSC_VER
#define WD_MODEL_ONNX	L"../../models/wd-vit-tagger-v2/model.onnx"
#define WD_LABEL_FNAME	"../../models/wd-vit-tagger-v2/selected_tags.csv"
#else
#define WD_MODEL_ONNX	"../models/wd-vit-tagger-v2/model.onnx"
#define WD_LABEL_FNAME	"../models/wd-vit-tagger-v2/selected_tags.csv"
#endif

// タグファイルに書かれているカテゴリーの意味
#define WD_TAG_CATEGORY_RATING	9				// レーティング用タグ
#define WD_TAG_CATEGORY_GENERAL	0				// 一般タグ
#define WD_TAG_CATEGORY_CHARA	4				// キャラクター用タグ

// ラベル一つを表すためのクラス
class TaggerLabel
{
public:
	std::string		name;				// ラベル名
	int				category;			// カテゴリ
	float			score;				// 推論した結果を入れるためのスコア

public:
	TaggerLabel(const std::string &sname, const std::string scategory_str){
		name = sname;
		category = ::atoi(scategory_str.c_str());
		score = 0;
	}
};

typedef std::vector<TaggerLabel>			TaggerLabelVec;
typedef std::vector<const TaggerLabel*>		TaggerLabelPtrVec;


// delimに指定した文字でトークンを切り出して結果を返す
std::vector<std::string> StringTokenize(const std::string& src, char delim)
{
    std::istringstream ss(src);
    std::string token;
	std::vector<std::string> dests;

    while (std::getline(ss, token, delim)) {
		dests.push_back(token);
    }
	return dests;
}

// ラベルデータの読み込みを行う
bool loadlabel(TaggerLabelVec &master, TaggerLabelPtrVec& ratings, TaggerLabelPtrVec& generals, TaggerLabelPtrVec& charas)
{
    std::ifstream labelfile(WD_LABEL_FNAME);

	std::cout << "read label file..." << std::endl;

	if(! labelfile){
		std::cerr << "Error: can not open label file" << std::endl;
		return false;
	}

    std::string line;
	std::getline(labelfile, line);				// 最初の行は読み飛ばし
    while (std::getline(labelfile, line)) {  	// 1行ずつ読み込む
		std::vector<std::string> tokens = StringTokenize(line, ',');
		if(tokens.size() < 3){
			std::cerr << "Error: label data is invarid format" << std::endl;
			return false;
		}
		// id,名前,カテゴリー,カウント数の順番
		master.push_back(TaggerLabel(tokens[1], tokens[2]));
	}

	// カテゴリにしたがって割り振る
	for(const auto& tag: master){
		switch(tag.category){
			case WD_TAG_CATEGORY_RATING:
				ratings.push_back(&tag);
				break;
			case WD_TAG_CATEGORY_CHARA:
				charas.push_back(&tag);
				break;
			case WD_TAG_CATEGORY_GENERAL:
				generals.push_back(&tag);
				break;
		}
    }

	std::cout << "read label ok." << std::endl;
	return true;
}

int main(int argc, char** argv )
{
	if (argc != 2){
		std::cerr << "usage: wdtagger <Image_Path>" << std::endl;
		return -1;
	}
	// イメージをファイルから読む
	cv::Mat src_image;
	src_image = cv::imread(argv[1], 1);
	if (! src_image.data){
		std::cerr << "cv::imread Error: can not read image" << std::endl;
		return -1;
	}
	// TODO: チャンネルが3以外は今回パス
	if (src_image.channels() != 3) {
		std::cerr << "sorry this program color channel 3 image only..." << std::endl;
		return -1;
	}

	TaggerLabelVec master;							// ラベルデータのマスター
	TaggerLabelPtrVec ratings, generals, charas;	// それぞれレーティング、一般タグ、キャラ名タグ。masterへのポインタが入っている
	// ラベルファイルを読む
	if(! loadlabel(master, ratings, generals, charas)){
		return -1;
	}

	Ort::AllocatorWithDefaultOptions ortallocator;
	Ort::Session ortsession{nullptr};
	Ort::SessionOptions sessionOptions;
	Ort::Env ortenv(ORT_LOGGING_LEVEL_WARNING,"wdtagger");
	Ort::MemoryInfo ortmem(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU));

	// Onnxモデルのロード
	std::cout << "loading wd-tagger model... " << std::endl;
	try{
		ortsession = Ort::Session(ortenv, WD_MODEL_ONNX, sessionOptions);
	}catch(Ort::Exception& e) {
		std::cerr << "ort::session Error:" << e.what() << std::endl;
	}
	std::cout << "load wd-tagger model success" << std::endl;

	// モデルが要求する配列形状などをとってくる
	std::vector<int64_t> input_shapes = ortsession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> output_shapes = ortsession.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	cv::Size model_insize;
	model_insize.width = (int)input_shapes[1];		// batch,width,height,channelの順番
	model_insize.height = (int)input_shapes[2];
	int	model_outsize = (int)output_shapes[1];		// batch,出力サイズの順番

	// 一応モデルの出力とラベルファイルの項目サイズが一致しているかを調べる
	if(master.size() != model_outsize){
		std::cerr << "Error : model size is not equal label data" << std::endl;
		return -1;
	}

	// モデルへの入出力データの確保
	std::vector<float> model_input_data(1 * model_insize.width * model_insize.height *3);
	std::vector<float> model_output_data(1 * model_outsize);

	// そのままではダメなのでモデルが受け付けるイメージへリサイズ
	// TODO: ここは単純にするために無理やりmodelの入力サイズに合わせているけど
	//       元ソースのように真ん中に画像を配置したほうがいいかもしれない
	cv::Mat tgt_image;
    cv::resize(src_image,tgt_image, model_insize, 1, 1, cv::INTER_CUBIC);
	if(! tgt_image.isContinuous()){		// 念の為
		std::cerr << "Error : cv::mat is not memory continuous..." << std::endl;
		return -1;
	}
//	cv::imwrite("./temp.jpg",tgt_image);		// TEST

	// 浮動小数点型へ直す
	{
		unsigned char* isrc = tgt_image.data;
		// NOTE: wd-taggerのモデルはBGRでトレーニングされているらしい
		//       OpenCVのMatはBGRで格納されているのでそのまま渡すことが出来るが
		//       OpenCV以外を使う場合は気をつけないとマズい
		for (auto &idata : model_input_data){
			idata = static_cast<float>(*isrc);
			isrc++;
		}
	}

	//入出力用のテンソルを作成する
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(ortmem, model_input_data.data(), model_input_data.size(), input_shapes.data(), input_shapes.size());
	Ort::Value output_tensor = Ort::Value::CreateTensor<float>(ortmem, model_output_data.data(), model_output_data.size(), output_shapes.data(), output_shapes.size());

	// モデルの推論実行
	std::cout << "Running predict..." << std::endl;
	try {
		std::vector<char const *> input_node_names;
		std::vector<char const *> output_node_names;
		std::string istr = ortsession.GetInputNameAllocated(0, ortallocator).get();
		std::string ostr = ortsession.GetOutputNameAllocated(0, ortallocator).get();
		input_node_names.push_back(istr.c_str());
		output_node_names.push_back(ostr.c_str());
		
		ortsession.Run(
			Ort::RunOptions{nullptr},
			input_node_names.data(),
			&input_tensor,
			1,
			output_node_names.data(),
			&output_tensor,
			1
		);
	}catch(Ort::Exception& e) {
		std::cerr << "ort::session::Run Error:" << e.what() << std::endl;
		return -1;
	}catch(...){
		std::cerr << "ort::session::Run : unknown Error" << std::endl;
		return -1;
	}
	std::cout << "predict success." << std::endl;

	// ラベルのマスタデータに結果を反映
	for(size_t i = 0; i < model_output_data.size(); i++){
		master[i].score = model_output_data[i];
	}
	std::cout << "result -----  " << std::endl;

	// それそれ推論結果を表示する
	std::cout << "ratings: " << std::endl;
	for(const auto& rt: ratings){
		std::cout << "   " << rt->name << ": " << rt->score << std::endl;
	}

	std::cout << "tags: " << std::endl;
	// generalタグは頻出度でソート
    std::sort(generals.begin(), generals.end(), [](const TaggerLabel* a, const TaggerLabel* b) {
        return a->score > b->score;
    });	
	for(size_t i = 0; i < 10; i++){
		std::cout << "   " << generals[i]->name << ": " << generals[i]->score << std::endl;
	}

	std::cout << "charas: " << std::endl;
	// 頻出度でソート
    std::sort(charas.begin(), charas.end(), [](const TaggerLabel* a, const TaggerLabel* b) {
        return a->score > b->score;
    });	
	for(size_t i = 0; i < 10; i++){
		std::cout << "   " << charas[i]->name << ": " << charas[i]->score << std::endl;
	}

	return 0;
}

