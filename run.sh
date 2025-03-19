NUM_ATTEMPTS=10


#./build/Yolov10ONNX models/s/yolov10s_dyn.onnx $NUM_ATTEMPTS $NUM_THREADS
#./build/Yolov10ONNX models/m/yolov10m_dyn.onnx $NUM_ATTEMPTS $NUM_THREADS
#./build/Yolov10ONNX models/b/yolov10b_dyn.onnx $NUM_ATTEMPTS $NUM_THREADS
#./build/Yolov10ONNX models/l/yolov10l_dyn.onnx $NUM_ATTEMPTS $NUM_THREADS

#./build/Yolov10TRT models/s/yolov10s_dyn.onnx $NUM_ATTEMPTS 
#./build/Yolov10TRT models/m/yolov10m_dyn.onnx $NUM_ATTEMPTS 
#./build/Yolov10TRT models/b/yolov10b_dyn.onnx $NUM_ATTEMPTS 
#./build/Yolov10TRT models/l/yolov10l_dyn.onnx $NUM_ATTEMPTS 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/
cd python/
#python yolov10trt.py --model_path ../models/m/yolov10m_dyn.onnx
#python yolov10trt.py --model_path ../models/b/yolov10b_dyn.onnx
#spython yolov10trt.py --model_path ../models/l/yolov10l_dyn.onnx

NUM_THREADS=-1
python yolov10onnx.py --model_path ../models/s/yolov10s_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/m/yolov10m_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/b/yolov10b_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/l/yolov10l_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS

NUM_THREADS=1
python yolov10onnx.py --model_path ../models/s/yolov10s_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/m/yolov10m_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/b/yolov10b_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/l/yolov10l_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS

NUM_THREADS=4
python yolov10onnx.py --model_path ../models/s/yolov10s_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/m/yolov10m_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/b/yolov10b_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/l/yolov10l_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS

NUM_THREADS=8
python yolov10onnx.py --model_path ../models/s/yolov10s_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/m/yolov10m_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/b/yolov10b_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/l/yolov10l_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS

NUM_THREADS=16
python yolov10onnx.py --model_path ../models/s/yolov10s_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/m/yolov10m_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/b/yolov10b_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS
python yolov10onnx.py --model_path ../models/l/yolov10l_dyn.onnx --num_attempts $NUM_ATTEMPTS --num_threads $NUM_THREADS