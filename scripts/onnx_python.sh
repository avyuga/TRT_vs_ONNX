NUM_ATTEMPTS=10

MODELS=(s m b l)
NUM_THREADS_LIST=(-1 1 8)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/

cd python/
for num_threads in "${NUM_THREADS_LIST[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "Running $model-model with $num_threads threads"
      python yolov10onnx.py \
          --model_path ../models/$model/yolov10${model}_dyn.onnx \
          --num_attempts $NUM_ATTEMPTS \
          --num_threads $num_threads
    done
done