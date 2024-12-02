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
    qwen = QwenObjectProposer("/root/autodl-tmp/Qwen2-VL-7B-Instruct", device)
    gsam = GroundedSAM(
        qwen=qwen,
        grounding_model_id="IDEA-Research/grounding-dino-base",
        sam2_checkpoint="./sam_ckpt/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    for video_idx in range(1, 10):
        json_path = f"/root/autodl-tmp/examples/{video_idx}/meta.json"
        left_video_path = f"/root/autodl-tmp/examples/{video_idx}/left.mp4"
        right_video_path = f"/root/autodl-tmp/examples/{video_idx}/right.mp4"

        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            instruction = data["current_task"]
            print(f"inst{video_idx}: {instruction}")

        torch.cuda.empty_cache()
        object, command = qwen.get_object_proposal_from_video(
            left_video_path = left_video_path,
            right_video_path = right_video_path,
            instruction = instruction,
        )
        print(f"command{video_idx}: {command}")

        torch.cuda.empty_cache()
        video = open_video(left_video_path)
        masks = gsam.get_video_mask(
            video=video,
            text=object,
            visualize=True,
            visualize_fname=f"track_{video_idx}_left.mp4",
            initial_threshold=0.05,
        )

        torch.cuda.empty_cache()
        video = open_video(right_video_path)
        masks = gsam.get_video_mask(
            video=video,
            text=object,
            visualize=True,
            visualize_fname=f"track_{video_idx}_right.mp4",
            initial_threshold=0.05,
        )
    import pdb; pdb.set_trace()