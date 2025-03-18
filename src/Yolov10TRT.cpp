#include "Yolov10TRT.h"
#include "Timers.h"

void Logger::log(Severity severity, const char* msg) noexcept
{
    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}


bool Yolov10TRT::build(){

    std::ifstream file(mParams.engineFileName, std::ios::binary);
    if (file.good()){
        std::cout << "Engine file with such name `" << mParams.engineFileName << "` already exists, exiting." << std::endl;
        return true;
    }
    
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(this->logger));
    assert(builder != nullptr);
    
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    assert(network != nullptr);

    
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    assert(config != nullptr);

    if (mParams.fp16) config->setFlag(BuilderFlag::kFP16);
    if (mParams.bf16) config->setFlag(BuilderFlag::kBF16);
    if (mParams.int8){
        config->setFlag(BuilderFlag::kINT8);
        setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }

    enableDLA(builder.get(), config.get(), mParams.dlaCore);

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, this->logger));
    assert(parser != nullptr);
    
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(parsed);

    const auto input = network->getInput(0);
    const auto inputName = input->getName();

    // Specify the optimization profile
    nvinfer1::IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(8, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(16, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    config->addOptimizationProfile(optProfile);

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    auto cudaStreamErrorCode = cudaStreamCreate(&profileStream);
    assert(cudaStreamErrorCode == 0);
    
    config->setProfileStream(profileStream);

    std::unique_ptr<IHostMemory> plan {builder->buildSerializedNetwork(*network, *config)};
    assert(plan != nullptr);
    

    auto runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(this->logger), InferDeleter());
    assert(runtime != nullptr);


    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    assert(engine != nullptr);


    // save engine to binary file
    std::ofstream outfile(mParams.engineFileName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Success, saved engine to " << mParams.engineFileName << std::endl;
    cudaStreamDestroy(profileStream);

    return true;
}


bool Yolov10TRT::load(){
    std::vector<char> trtModelStream_;
    size_t size{0};
    
    std::ifstream file(mParams.engineFileName, std::ios::binary);
    assert(file.good());
    
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
        
    trtModelStream_.resize(size);
    file.read(trtModelStream_.data(), size);
    file.close();
    
    
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->logger));
    assert(mRuntime != nullptr);
    
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(trtModelStream_.data(), size));
    assert(mEngine != nullptr);
    
    return true;
}


void Yolov10TRT::detect(
    std::vector<cimg_library::CImg<float>> imgList, 
    float* rawOutput
){
    
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    assert(context);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvinfer1::Dims4 inputDims = {imgList.size(), mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth};
    context->setInputShape(mParams.inputTensorNames[0].c_str(), inputDims);

    BufferManager buffers(mEngine, imgList.size(), context.get());

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }
    assert(mParams.inputTensorNames.size() == 1); // only one model entrance

    
    // copy img from CImg instance to HostBuffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i=0; i<imgList.size(); ++i){
        auto img = imgList[i];
        std::copy(img.data(), img.data() + img.size(), hostDataBuffer + i*img.size());
    }

    // Memcpy from host input buffers to device input buffers
    // buffers.copyInputToDevice();
    buffers.copyInputToDeviceAsync(stream);

    // bool status = context->executeV2(buffers.getDeviceBindings().data());
    bool status = context->enqueueV3(stream);
    assert(status);


    // Memcpy from device output buffers to host output buffers
    // buffers.copyOutputToHost();
    buffers.copyOutputToHostAsync(stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    std::copy(output, output + imgList.size()*mParams.outputLength*mParams.outputItemSize*sizeof(float), rawOutput);

}



int main(int argc, char** argv)
{
    assert(argc == 3);
    char* onnxFileName = argv[1];
    char* numInferenceAttempts = argv[2];

    std::filesystem::path onnxFilePath(onnxFileName);
    std::string engineFileName = onnxFilePath.replace_extension("engine").string();
  
    Params params;
    
    params.onnxFileName = onnxFileName;
    params.engineFileName = engineFileName.c_str();
    
    params.inputTensorNames.push_back("images");
    params.outputTensorNames.push_back("output0");
    params.dlaCore = -1; // not supported on the server
    params.int8 = false;
    params.fp16 = false;
    params.bf16 = false;

    params.inputHeight = 640;
    params.inputWidth = 640;
    params.inputNChannels = 3;

    params.outputLength = 300;
    params.outputItemSize = 6;

    params.numInferenceAttempts = std::stoi(numInferenceAttempts);
    params.outputFileName = "cpp_result.csv";
    
    Yolov10TRT Engine(params);

    std::cout << "Building and running a GPU inference engine for " << onnxFileName << std::endl;
    auto status = Engine.build();
    std::cout << std::boolalpha << "Build Engine with status " << status << std::endl;
    
    status = Engine.load();
    std::cout << std::boolalpha << "Load Engine with status " << status << std::endl;
    
    const std::filesystem::path img_path{"assets/"};
    std::vector<cimg_library::CImg<float>> fullImgList = Utility::processInput(params, img_path);
    int numberOfImages = fullImgList.size();
    std::cout << "Number of Images: " << numberOfImages << std::endl;
    std::cout << std::endl;

    std::mt19937 randomRange(std::random_device{}());
    Timer timer;

    std::vector<double> times(numberOfImages);

    for (int batchSize = 1; batchSize <= numberOfImages; ++batchSize){
        std::vector<cimg_library::CImg<float>> randomBatch(batchSize);
        float* rawOutput = new float[batchSize * params.outputItemSize * params.outputLength * sizeof(float)];

        std::vector<double> timePerBatch;
        for (int attempt=0; attempt<params.numInferenceAttempts; ++attempt){
            std::sample(fullImgList.begin(), fullImgList.end(),
                randomBatch.begin(), batchSize, randomRange);
            timer.tic();
            Engine.detect(randomBatch, rawOutput);
            double diff = timer.toc();
            timePerBatch.push_back(diff);
        }
        
        double meanTimePerBatch = accumulate(timePerBatch.begin(), timePerBatch.end(), 0) / timePerBatch.size();
        std::cout << "Batch size=" << batchSize << " took " << meanTimePerBatch  << " ms, "  <<
            meanTimePerBatch/batchSize << " ms/img" << std::endl;
        times[batchSize-1] = meanTimePerBatch / batchSize;

        if (batchSize == numberOfImages){
            std::vector<std::vector<Detection>> resultList = Utility::processOutput(rawOutput, batchSize, params);
            
            assert(batchSize == resultList.size());
            for(int i = 0; i < batchSize; ++i){
                auto img = fullImgList[i];
                auto result = resultList[i];
    
                std::string filename = "results/trt/" + std::to_string(i) + ".png";
                Utility::drawResult(img, result, filename.c_str());
            }
        }

        delete[] rawOutput;
    }

    Utility::logInference(params, "TRT", fullImgList.size(), times);
    return 0;
}