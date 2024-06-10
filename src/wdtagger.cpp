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
#include <opencv2/opencv.hpp>
#include "onnx/onnxruntime_cxx_api.h"

#define WD_MODEL_ONNX "../models/wd-vit-tagger-v2/model.onnx"
#define WD_TAG_FNAME "../models/wd-vit-tagger-v2/tagger.csv"

int main(int argc, char** argv )
{
	if ( argc != 2 ){
		std::cerr << "usage: wdtagger <Image_Path>" << std::endl;
		return -1;
	}
	// イメージをファイルから読む
	cv::Mat src_image;
	src_image = cv::imread( argv[1], 1 );
	if ( !src_image.data ){
		std::cerr << "cv::imread Error: can not read image" << std::endl;
		return -1;
	}
	// TODO: チャンネルが3以外は今回パス
	if (src_image.channels() != 3) {
		std::cerr << "sorry this program color channel 3 image only..." << std::endl;
		return -1;
	}

//	std::string vstr = Ort::GetVersionString();
//	std::cout << "onnxver" << vstr << std::endl;

	std::unique_ptr<Ort::Env> ortenv;
	std::unique_ptr<Ort::MemoryInfo> ortmem;
	Ort::AllocatorWithDefaultOptions ortallocator;
	Ort::Session ortsession{nullptr};

	Ort::SessionOptions sessionOptions;
	ortenv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "wdtagger");
	ortmem = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU));

	// Onnxモデルのロード
	std::cout << "loading wd-tagger model... " << std::endl;
	try{
		ortsession = Ort::Session(*ortenv, WD_MODEL_ONNX, sessionOptions);
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

	// モデルへの入出力データの確保
	std::vector<float> model_input_data(1 * model_insize.width * model_insize.height *3);
	std::vector<float> model_output_data(1 * model_outsize);

	// そのままではダメなのでモデルが受け付けるイメージへリサイズ
	// TODO: ここは単純にするために無理やりmodelの入力サイズに合わせているけど
	//       元ソースのように真ん中に画像を配置したほうがいいかもしれない
	cv::Mat tgt_image;
    cv::resize(src_image,tgt_image, model_insize, 1 , 1, cv::INTER_CUBIC);
//	cv::imwrite("./temp.jpg",tgt_image);		//TEST

	// 浮動小数点型へ直す
	{
		size_t mx = model_insize.width * model_insize.height * 3;
		float *mdata = model_input_data.data();
		unsigned char* src = tgt_image.data;
		// NOTE: wd-taggerのモデルはBGRでトレーニングされているらしい
		//       なので、OpenCVはBGRで格納されているのでそのまま渡しても良い
		//       OpenCV以外を使う場合は気をつけないとマズい
		for(size_t i=0; i < mx;i+=3){
			mdata[i] = static_cast<float>(src[i]);
			mdata[i+1] = static_cast<float>(src[i+1]);
			mdata[i+2] = static_cast<float>(src[i+2]);
		}
	}

	//入出力用のテンソルを作成する
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(*ortmem, model_input_data.data(), model_input_data.size(), input_shapes.data(), input_shapes.size());
	Ort::Value output_tensor = Ort::Value::CreateTensor<float>(*ortmem, model_output_data.data(), model_output_data.size(), output_shapes.data(), output_shapes.size());

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
	return 0;
}

