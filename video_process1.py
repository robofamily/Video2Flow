import json
import av
from PIL import Image
import numpy as np
import torch
from qwenvl import QwenObjectProposer
from grounded_sam2_hf_model import GroundedSAM
from tqdm import tqdm

def open_video(video_path):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frame_np = frame.to_ndarray(format='rgb24')
        frames.append(frame_np)
    video_array = np.array(frames)
    container.close()
    return video_array

if __name__ == '__main__':
    device = torch.device("cuda")
    qwen = QwenObjectProposer("Qwen/Qwen2-VL-2B-Instruct", device)
    gsam = GroundedSAM(
        qwen=qwen,
        grounding_model_id="IDEA-Research/grounding-dino-base",
        sam2_checkpoint="../sam2/checkpoints/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    video = open_video("./test.mp4")
    object = qwen.get_object_proposal_from_video(
        video_path = "./test.mp4",
        instruction="Move object into or out of container (ex: drawer, clothes hamper, plate, trashcan, washer)",
    )
    import pdb; pdb; pdb.set_trace()
    masks = gsam.get_video_mask(
        video = video,
        text=object,
        visualize=True,
        visualize_fname=f"track.mp4",
        initial_threshold=0.05,
    )
    import pdb; pdb.set_trace()