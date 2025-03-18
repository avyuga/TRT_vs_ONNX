cd ~/AIArchitecture/tensorrt
case $1 in
    "rebuild")
        rm -rf build/*
        cmake . -B build/
        cmake --build build/ ;;
    "clean")
        rm -rf build/* ;;
esac
