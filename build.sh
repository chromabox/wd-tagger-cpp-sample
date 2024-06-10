#!/bin/bash

check_and_create_dir()
{
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
}

build_debug()
{
    check_and_create_dir build
    cd build

    cmake -DCMAKE_BUILD_TYPE=Debug ..
    cmake --build .

    cd ../
}

build_release()
{
    check_and_create_dir build_release
    cd build_release

    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build .

    cd ../
}


clean_build()
{
    rm -rf build
    rm -rf build_release
}

if [ "${1}" = "--clean" ]; then
    clean_build
    exit 0
fi

check_and_create_dir models/wd-vit-tagger-v2
check_and_create_dir onnxlib

if [ ! -e "onnxlib/onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so" ]; then
    echo "onnxruntime is not found.. download github..."
    cd onnxlib
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
    tar zxvf onnxruntime-linux-x64-1.18.0.tgz
    rm -rf onnxruntime-linux-x64-1.18.0.tgz
    cd ..
fi

if [ ! -e "models/wd-vit-tagger-v2/model.onnx" ]; then
    echo "wd-vit model is not found.. download hugging face..."
    cd models/wd-vit-tagger-v2
    wget https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/model.onnx
    wget https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/raw/main/selected_tags.csv
    cd ../../
fi

# オプション指定の振り分け
if [ "${1}" = "--release" ]; then
    build_release
    echo "done."
    exit 0
fi

build_debug

echo "done."
