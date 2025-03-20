import numpy as np
import random
import time
from utils import preprocess_all_images, Detection, BBox, parse_args, log_inference

import onnxruntime as ort


class YoloV10ONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': 0}]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        
    def detect(self, batch: np.ndarray) -> np.ndarray:
        result = self.session.run([self.output_name], {self.input_name: batch.astype(np.float32)})[0]
        return result

    
if __name__ == "__main__":
    args = parse_args()
    onnx_model = YoloV10ONNX(args.model_path)
    
    image_batch = preprocess_all_images("../assets")
    batch = np.stack(image_batch, axis=0).astype(np.float32)

    times = []
    for batch_size in range(1, batch.shape[0]+1):
        ms_time = []
        for j in range(args.num_attempts):
            random_idxs = random.sample(range(batch.shape[0]), batch_size)
            batch_sample = batch[random_idxs]
            t1 = time.time()
            output = onnx_model.detect(batch_sample)
            t2 = time.time()
            ms_time += [(t2-t1)*1000]
        print(f"Batch size={batch_size} took {np.mean(ms_time):.3f} ms, {np.mean(ms_time)/batch_size:.3f} ms/img")
        times += [np.mean(ms_time)/batch_size]
    
    log_inference(args, "ONNX", batch.shape[0], times)

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
    
    