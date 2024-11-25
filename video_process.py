import json
import av
from PIL import Image
import numpy as np
import torch
from qwenvl import QwenObjectProposer
from grounded_sam2_hf_model import GroundedSAM

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
        sam2_checkpoint="../Grounded-SAM-2/checkpoints/sam2_hiera_large.pt",
        sam2_config="sam2_hiera_l.yaml",
    )
    meta_data = json.load(open("../../droid_low_resolution/index.json", "r"))
    video_idx = 1000
    video = open_video("../../droid_low_resolution/" + meta_data[video_idx]['path'])
    object = qwen.get_object_proposal(
        image = video[0],
        instruction=meta_data[video_idx]['cap'][0],
    )
    masks = gsam.get_video_mask(
        video = video,
        text=object,
        visualize=True,
        initial_threshold=0.05,
    )
    import pdb; pdb.set_trace()