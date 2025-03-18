#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <list>
#include <vector>
#include <random>
#include <numeric>
#include <stdexcept>

#include "Utils.h"
#include "Buffers.h"

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override;
};


struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};


inline void enableDLA(
    nvinfer1::IBuilder* builder, 
    nvinfer1::IBuilderConfig* config, 
    int useDLACore, 
    bool allowGPUFallback = true
){
    if (useDLACore >= 0){
        if (builder->getNbDLACores() == 0) {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            throw std::runtime_error("Error: use DLA core on a platfrom that doesn't have any DLA cores");
        }
        if (allowGPUFallback) {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)) {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

inline void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange = 2.0F, float outRange = 4.0F)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                assert(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    assert(output->setDynamicRange(-inRange, inRange));
                }
                else
                {
                    assert(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}

class Yolov10TRT{
    public:
        Yolov10TRT(Params& params): 
            mParams(params),
            mRuntime(nullptr),
            mEngine(nullptr){
        }
    
        bool build();
        bool load();
        // std::list<Detection> detect(cimg_library::CImg<float> img);
        void detect(
            std::vector<cimg_library::CImg<float>> img_list, 
            float* rawOutput
        );
        
    
    private:
        Params mParams; // The parameters for the sample.
    
        nvinfer1::Dims mInputDims;  // The dimensions of the input to the network.
        nvinfer1::Dims mOutputDims; // The dimensions of the output to the network.
        int mNumber{0};             // The number to classify
    
        std::shared_ptr<nvinfer1::IRuntime> mRuntime;   // The TensorRT runtime used to deserialize the engine
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine; // The TensorRT engine used to run the network
        std::vector<std::vector<Detection>> processOutput(float* output, int numImages, Params params);
    
        Logger logger;
};