import os
import random
import time

import common
import numpy as np
from cuda import cudart

from utils import (BBox, Detection, log_inference, parse_args,
                   preprocess_all_images)

import tensorrt as trt
TRT_LOGGER = trt.Logger()


class YoloV10TRT:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.engine_path = onnx_model_path[:-4] + "engine"
        assert(os.path.exists(self.engine_path))
        self.load()
        
    def load(self):
        # предполагается, что файл .engine уже был собран кодом на c++
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert(self.engine is not None)
        print(f"Successfully loaded from {self.engine_path}")

    '''
    Используется кастомная аллокация буферов, поскольку вариант, предлагаемый авторами, использует оптимизационный профиль и на тензоре-выводе, что приводит к неверным размерностям
    '''
    def _allocate_buffers(self, current_batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = common.cuda_call(cudart.cudaStreamCreate())
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        
        for binding in tensor_names:
            shape = self.engine.get_tensor_shape(binding)
            shape[0] = current_batch_size
            
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
                
            size = trt.volume(shape)
            trt_type = self.engine.get_tensor_dtype(binding)
    
            # Allocate host and device buffers
            if trt.nptype(trt_type):
                dtype = np.dtype(trt.nptype(trt_type))
                bindingMemory = common.HostDeviceMem(size, dtype)
            else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
                size = int(size * trt_type.itemsize)
                bindingMemory = common.HostDeviceMem(size)
    
            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))
    
            # Append to the appropriate list.
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(bindingMemory)
            else:
                outputs.append(bindingMemory)
        return inputs, outputs, bindings, stream

        
    def detect(self, batch: np.ndarray):
        with self.engine.create_execution_context() as context:
            context.set_input_shape("images", (batch.shape[0], 3, 640, 640))
            
            inputs, outputs, bindings, stream = self._allocate_buffers(batch.shape[0])
            inputs[0].host = batch
            
            trt_outputs = common.do_inference(
                context,
                engine=self.engine,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
        trt_outputs = trt_outputs[0].reshape(batch.shape[0], 300, 6)
        return trt_outputs
    

if __name__ == "__main__":
    args = parse_args()
    
    trt_model = YoloV10TRT(args.model_path)

    image_batch = preprocess_all_images("../assets")
    batch = np.stack(image_batch, axis=0).astype(np.float32)

    times = []
    for batch_size in range(1, batch.shape[0]+1):
        ms_time = []
        for j in range(args.num_attempts):
            random_idxs = random.sample(range(batch.shape[0]), batch_size)
            batch_sample = batch[random_idxs]

            t1 = time.time()
            trt_output = trt_model.detect(batch_sample)
            t2 = time.time()
            ms_time += [(t2-t1)*1000]

        print(f"Batch size={batch_size} took {np.mean(ms_time):.3f} ms, {np.mean(ms_time)/batch_size:.3f} ms/img")
        times += [np.mean(ms_time)/batch_size]
    
    log_inference(args, "TRT", batch.shape[0], times)

    '''
    results = []
    for i in range(batch.shape[0]):
        for j in range(300):
            row = trt_output[i, j]
            if row[4] < 0.6: continue;
            results.append(
                Detection(
                    bbox=BBox(row[0], row[1], row[2], row[3]), 
                    score=row[4], 
                    cls_id=row[5]
                )
            )
            
    for r in results:
        print(r)
    
    '''
    
    