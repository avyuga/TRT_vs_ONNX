NUM_ATTEMPTS=10

MODELS=(s m b l)
#NUM_THREADS_LIST=(-1 1 4 8 16)
NUM_THREADS_LIST=(-1)


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jovyan/workspace/cpp_libs/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/lib/:/home/jovyan/workspace/cpp_libs/onnxruntime-linux-x64-gpu-1.20.0/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/

case $1 in
    "TRT_cpp")
        for model in "${MODELS[@]}"; do            
          ./build/Yolov10TRT models/$model/yolov10${model}_dyn.onnx $NUM_ATTEMPTS
        done
        ;;
    "ONNX_cpp")
        for num_threads in "${NUM_THREADS_LIST[@]}"; do
            for model in "${MODELS[@]}"; do
              ./build/Yolov10ONNX models/$model/yolov10${model}_dyn.onnx $NUM_ATTEMPTS $num_threads
            done
        done
        ;;
    "TRT_python")
        cd python/
        for model in "${MODELS[@]}"; do  
            python yolov10trt.py --model_path ../models/$model/yolov10${model}_dyn.onnx --num_attempts $NUM_ATTEMPTS
        done
        ;;
    "ONNX_python")
        # может выдавать проблемы с LD_LIBRARY_PATH, работает скрипт onnx_python.sh
        cd python/
        for num_threads in "${NUM_THREADS_LIST[@]}"; do
            for model in "${MODELS[@]}"; do
              python yolov10onnx.py \
                  --model_path ../models/$model/yolov10${model}_dyn.onnx \
                  --num_attempts $NUM_ATTEMPTS \
                  --num_threads $num_threads
            done
        done
        ;;
esac