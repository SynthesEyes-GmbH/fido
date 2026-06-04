import numpy as np
import torch


def load_model(model_path):
    print(f"Loading model from {model_path}...")
    print(f"Cuda is available: {torch.cuda.is_available()}")
    return


def inference(task_id, oct_data, opmi_image, model):

    if task_id == 0:
        return {
            'keypoints': np.random.rand(2).tolist(),
            'tool_tissue_distance': float(np.random.rand())
        }

    if task_id == 1:
        return np.random.rand(3, 3)
    
    else:
        return None
