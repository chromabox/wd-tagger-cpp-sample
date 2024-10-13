Param(
	[parameter(mandatory=$true)][String]$switcher
)

function check_and_create_dir($dirname){
    if( -not ( Test-Path $dirname ) ){
        New-Item $dirname -ItemType Directory -ErrorAction SilentlyContinue
    }
}

function build_debug()
{
    cd build
    cmake ..
    cmake --build .
    cp ../onnxlib/onnxruntime-win-x64-1.18.0/lib/*.dll Debug/
    cd ../
}

function clean_build()
{
    Remove-Item -Force -Recurse build
    Remove-Item -Force -Recurse build_release
}

check_and_create_dir build
check_and_create_dir models/wd-vit-tagger-v2
check_and_create_dir onnxlib

if ( -not (Test-Path "onnxlib/onnxruntime-win-x64-1.18.0/lib") ){
    Write-Host "onnxruntime is not found.. download github..."
    cd onnxlib
    curl.exe -L "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-1.18.0.zip" --output "onnxruntime.zip"
    tar zxvf onnxruntime.zip
    Remove-Item -Force onnxruntime.zip
    cd ..
}

if ( -not (Test-Path "models/wd-vit-tagger-v2/model.onnx") ){
    Write-Host "wd-vit model is not found.. download hugging face..."
    cd models/wd-vit-tagger-v2
    curl.exe -L "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/model.onnx" --output "model.onnx"
    curl.exe -L "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/raw/main/selected_tags.csv" --output "selected_tags.csv"
    cd ../../
}

&{Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"; Enter-VsDevShell b720724c -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"}

if ($switcher -eq "clean"){
    clean_build
    exit 0
}

build_debug

exit $LASTEXITCODE
