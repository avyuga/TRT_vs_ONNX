#include <stdint.h>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <regex>
#include <experimental/iterator>


#define cimg_use_png
#include "CImg.h"

#ifndef UTILS
#define UTILS


struct Params
{
    std::string onnxFileName;
    std::string engineFileName;
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    bool bf16{false};                  //!< Allow running the network in BF16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string timingCacheFile; //!< Path to timing cache file

    std::string saveEngine;
    std::string loadEngine;

    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t inputNChannels;

    uint32_t outputLength;
    uint32_t outputItemSize;

    int numThreads{-1};
    int numInferenceAttempts{5};
    std::string outputFileName;
};


class BBox{
    public:
        uint32_t x0;
        uint32_t y0;
        uint32_t x1;
        uint32_t y1;
        BBox(uint32_t x0_, uint32_t y0_, uint32_t x1_, uint32_t y1_):
            x0(x0_), y0(y0_), x1(x1_), y1(y1_) {
        }

};

class Detection{
    public: 
        int classId;
        float score;
        BBox bbox;

        Detection(BBox bbox_, float score_, int classId_):
            bbox(bbox_),
            score(score_),
            classId(classId){
        }
    
};

class Utility{
    public:
        static std::vector<cimg_library::CImg<float>> processInput(Params p, const std::filesystem::path img_folder);
        static void drawResult(
            cimg_library::CImg<float> img, 
            std::vector<Detection> detections, 
            const char* file_name
        );
        static std::vector<std::vector<Detection>> processOutput(float* output, int numImages, Params params);
        static void logInference(Params p, const char* engine, int maxBatchSize, std::vector<double> data);
};


#endif