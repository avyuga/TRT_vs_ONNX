#ifndef YOLOV10_ONNX
#define YOLOV10_ONNX

#include <string>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <random>

#include <list>
#include "CImg.h"
#include "Utils.h"
#include "Timers.h"



class Yolov10ONNX{
public:
    Yolov10ONNX(Params params);
    void detect(
        std::vector<cimg_library::CImg<float>> imgList, 
        float* rawOutput
    );
     

private:
    // ORT
    Params params;
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{ nullptr };
    Ort::Session session{ nullptr };

    // Names
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    // Shapes
    std::vector<int64_t> inputModelShape;
    std::vector<int64_t> outputModelShape;
    int64_t numInputElements;

};

#endif