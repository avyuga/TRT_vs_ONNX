import os
import numpy as np
from PIL import Image
import argparse
import re

### Классы, используемые для хранения найденных ббоксов и детекций
class BBox:
    def __init__(self, x0: int, y0: int, x1: int, y1: int):
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.x1 = int(x1)
        self.y1 = int(y1)

    def __str__(self):
        return f"Bbox with coords LT ({self.x0}, {self.y0}), RB ({self.x1}, {self.y1})"


class Detection:
    def __init__(self, cls_id: int, score: float, bbox: BBox):
        self.cls_id = cls_id
        self.score = score
        self.bbox = bbox

    def __str__(self):
        return f"Detection with cls={self.cls_id}[{self.score:.3f}] located at {self.bbox}"


### утилиты
def preprocess_all_images(directory: str) -> list:
    processed_images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            img = Image.open(file_path).convert('RGB')
            img_resized = img.resize((640, 640))
            img_array = np.array(img_resized)
            img_array = img_array.transpose(2, 0, 1)
            img_array = img_array / 255.0
            processed_images.append(img_array)
    return processed_images


<<<<<<< HEAD
def draw_on_image():
    pass


=======
>>>>>>> add python inference
def log_inference(args, engine: str, max_batch_size: int, data: np.ndarray):
    output_file_path = "../python_result.csv"
    
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w') as f:
<<<<<<< HEAD
            row = "Backend,Model,NumThreads," + \
                ",".join(map(str, list(range(1, max_batch_size+1)))) + "\n"
=======
            row = "Backend, Model, NumThreads, " + \
                ", ".join(map(str, list(range(1, max_batch_size+1)))) + "\n"
>>>>>>> add python inference
            f.write(row)
    with open(output_file_path, 'a') as f:
        onnx_file_name = args.model_path.split("/")[-1]
        pattern = r'v10([a-z]+)_dyn\.onnx'
        match = re.search(pattern, onnx_file_name)
        if match:
            model_variant = match.group(1)
        else:
            raise RuntimeError("Could not parse model name")
<<<<<<< HEAD
        row = f"{engine},{model_variant},{args.num_threads}," + ",".join(map(str, data)) + "\n"
=======
        row = f"{engine}, {model_variant}, {args.num_threads}, " + ", ".join(map(str, data)) + "\n"
>>>>>>> add python inference
        f.write(row)
        
    

def parse_args():
    parser = argparse.ArgumentParser(description='CLI Argumnets')
    parser.add_argument('--model_path', type=str, help='Path to onnx model')
    parser.add_argument('--num_attempts', type=int, default=5, help="Number of attempt sto evaluate")
    parser.add_argument('--num_threads', type=int, default=-1, help="Number of threads, default -1 - Auto. Applicable to ONNX only.")

    return parser.parse_args()





    