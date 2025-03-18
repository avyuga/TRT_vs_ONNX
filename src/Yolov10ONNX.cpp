#include "Yolov10ONNX.h"
#include <stdexcept>


Yolov10ONNX::Yolov10ONNX(Params params_){

	params = params_;
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "ONNX_DETECTION");
	sessionOptions = Ort::SessionOptions();	

	// сюда можно ставить число потоков, оптимальное для данного процессора
	sessionOptions.SetInterOpNumThreads(params.numThreads);
	sessionOptions.SetIntraOpNumThreads(params.numThreads);

	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();

    std::cout << "System available providers: ";
	for (const auto& provider : availableProviders) {
		std::cout << provider << " ";
	}
	std::cout << std::endl;
	
	auto cudaAvailable = find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");

	if (cudaAvailable != availableProviders.end()) {
		std::cout << "Inference provider: Cuda" << std::endl;
		
		OrtCUDAProviderOptions cudaOptions;
		cudaOptions.device_id = 0;
		cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
	} else {
		throw std::runtime_error("CUDAExecutionProvider is not available");
	}
	
	
    session = Ort::Session(env, params.onnxFileName.c_str(), sessionOptions);

	// -1 будет заменена на реальное значение в рантайме
	this->inputModelShape = {-1, 3, params.inputHeight, params.inputWidth};
	this->outputModelShape = {-1, params.outputLength, params.outputItemSize};

	// Print shapes
	std::cout << "Input shape: (";
	std::copy(this->inputModelShape.begin(), this->inputModelShape.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << ")\n";

	std::cout << "Output shape: (";
	std::copy(this->outputModelShape.begin(), this->outputModelShape.end(), std::ostream_iterator<int>(std::cout, ", "));
	std::cout << ")\n";

	// Get names
	Ort::AllocatorWithDefaultOptions allocator;

	Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, allocator);
	inputNames.push_back(inputName.get());
	inputName.release();
	std::cout << "Input name: " << inputNames[0] << std::endl;

	int64_t numberOfOutputs = session.GetOutputCount();
	for (int i = 0; i<numberOfOutputs; i++){
		Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(i, allocator);
		outputNames.push_back(outputName.get());
		outputName.release();
	}

	std::cout << "Output names: (";
	std::copy(this->outputNames.begin(), this->outputNames.end(), std::ostream_iterator<const char*>(std::cout, ", "));
	std::cout << ")\n" << std::endl;

};



void Yolov10ONNX::detect(
	std::vector<cimg_library::CImg<float>> imgList, 
	float* rawOutput
){
    // Define Tensors
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	this->inputModelShape[0] = imgList.size();
	this->outputModelShape[0] = imgList.size();

	numInputElements = accumulate(inputModelShape.begin(), inputModelShape.end(), sizeof(float), std::multiplies<>());
	
	float* imgDataTensor = new float[numInputElements];
	for (int i=0; i<imgList.size(); ++i){
        auto img = imgList[i];
        std::copy(img.data(), img.data() + img.size(), imgDataTensor + i*img.size());
    }

	std::vector<Ort::Value> inputTensors;
	inputTensors.push_back(
		Ort::Value::CreateTensor<float>(
			memoryInfo, imgDataTensor, numInputElements, inputModelShape.data(), inputModelShape.size()
		)
	);
	
	std::vector<Ort::Value> outputTensors = this->session.Run(
		Ort::RunOptions{ nullptr }, 
		inputNames.data(), inputTensors.data(), inputNames.size(), 
		outputNames.data(), outputNames.size()
	);

	auto* output = outputTensors[0].GetTensorData<float>(); // shape (N, 300, 6)
	std::copy(output, output + imgList.size()*params.outputLength*params.outputItemSize*sizeof(float), rawOutput);
	
};



int main(int argc, char** argv)
{

	assert(argc == 4);
    char* onnxFileName = argv[1];
	char* numInferenceAttempts = argv[2];
	char* numThreads = argv[3];

    Params params;
    
    params.onnxFileName = onnxFileName;
    params.inputHeight = 640;
    params.inputWidth = 640;
    params.inputNChannels = 3;

    params.outputLength = 300;
    params.outputItemSize = 6;

	params.numInferenceAttempts = std::stoi(numInferenceAttempts);
	params.numThreads = std::stoi(numThreads);

	params.outputFileName = "cpp_result.csv";


    Yolov10ONNX Engine(params);

    const std::filesystem::path img_path{"assets/"};
    std::vector<cimg_library::CImg<float>> img_list = Utility::processInput(params, img_path);
	std::vector<cimg_library::CImg<float>> fullImgList = Utility::processInput(params, img_path);
    int numberOfImages = fullImgList.size();
    std::cout << "Number of Images: " << numberOfImages << std::endl;
    std::cout << std::endl;

    std::mt19937 randomRange(std::random_device{}());
    Timer timer;

    std::vector<double> times;
    
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
        times.push_back(meanTimePerBatch / batchSize);

        if (batchSize == numberOfImages){
            std::vector<std::vector<Detection>> resultList = Utility::processOutput(rawOutput, batchSize, params);
            
            assert(batchSize == resultList.size());
            for(int i = 0; i < batchSize; ++i){
                auto img = fullImgList[i];
                auto result = resultList[i];
    
                std::string filename = "results/onnx/" + std::to_string(i) + ".png";
                Utility::drawResult(img, result, filename.c_str());
            }
        }

        delete[] rawOutput;

    }

	Utility::logInference(params, "ONNX", fullImgList.size(), times);
    return 0;
}