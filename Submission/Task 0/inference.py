import numpy as np
import torch


def load_model(model_path):
    print(f"Loading model from {model_path}...")
    print(f"Cuda is available: {torch.cuda.is_available()}")
    return


def inference(task_id, oct_data, opmi_image, model):

    return {
        'keypoints': np.random.rand(2).tolist(),
        'tool_tissue_distance': float(np.random.rand())
    }
