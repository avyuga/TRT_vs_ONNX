cd ~/workspace/AIArchitecture/TRT_vs_ONNX
case $1 in
    "rebuild")
        rm -rf build/*
        cmake . -B build/
        if [ -n "$2" ]; then
            cmake --build build/ --target "$2"
        else
            cmake --build build/
        fi
        ;;
    "clean")
        rm -rf build/* ;;
esac
