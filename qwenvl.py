import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def sample_frames(video_tensor, num_frames=6):
    T, C, H, W = video_tensor.size()
    if T < num_frames:
        raise ValueError("Number of video frames are fewer than num_frames")

    # Compute sampling interval
    step = T // num_frames

    # Sample frames uniformly
    indices = torch.arange(0, T, step)[:num_frames]
    sampled_video = video_tensor[indices]

    return sampled_video

class QwenObjectProposer:

    def __init__(self, model_id, device):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def infer(self, vision_content, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    vision_content,
                    {
                        "type": "text", 
                        "text": prompt,
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if video_inputs is not None:
            video_inputs = [sample_frames(video) for video in video_inputs]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_object_proposal(self, image, instruction):
        vision_content = {
            "type": "image",
            "image": Image.fromarray(image), 
        }
        prompt = "The instruction to the robot is: " \
        + instruction + \
        ". Please answer which object will be manipulated with a single word. " + \
        "For example, if the robotics arm will manipulate an apple, the answer should be: apple."
        object = self.infer(vision_content, prompt)[0] + '.'
        return object
        
    def get_object_proposal_from_video(self, video_path, instruction):
        vision_content = {
            "type": "video",
            "video": video_path, 
            "max_pixels": 1280 * 720,
            "fps": 60,
        }

        prompt_1 = "You are an operator who controls a robotics arm by language command. Please watch this demonstration video" + \
        "Now, the robotics arm is reset to the original position and you must generate suitable language instruction to repeat the manipulation in the video. " + \
        "A reference instruction to the robot is: " + instruction + ". However, the manipulated object may not be in the instruction and the instruction may be confusing. " + \
        "Your command should comtain 2 parts." + \
        "The first part is a short sentence that contains at least a manipulated object (word) and a manipulation (verb)." + \
        "The second part is a detailed description of the manipulation. " + \
        "Example command: [short] pick the apple and put it on the shelf. [long] approach the red apple on the right hand side of the desk. pick the red apple and then move to the black shelf in the room. place the apple on the top of the shelf."
        command = self.infer(vision_content, prompt_1)[0]

        prompt_2 = "Please watch this video and answer which object is grasped and manipulated by the robotics arm with a single word. " + \
        "The instruction to the robot is: " + command + \
        "For example, if the robotics arm will manipulate an apple, the answer should be: apple."
        object = self.infer(vision_content, prompt_2)[0] + '.'

        return object, command
    
    def verify_object(self, image, object):
        vision_content = {
            "type": "image",
            "image": Image.fromarray(image), 
        }
        prompt = "Is is a figure of " + object + "? Please answer with a single word. " + \
        "For example, if I ask you whether it is a figure of apple, and the main object of the figure is an apple, " + \
        "your answer should be: yes." + \
        "If I ask you whether it is a figure of apple, and the main object of the figure is a banana, " + \
        "your answer should be: no.",
        answer = self.infer(vision_content, prompt)[0]
        if 'yes' in answer:
            return True
        elif 'no' in answer:
            return False
        else:
            print("Invalid answer:", answer)
            return False

if __name__ == "__main__":
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    device = torch.device("cuda")
    qwen = QwenObjectProposer(model_id, device)
    object = qwen.get_object_proposal(
        image=np.array(Image.open("./test.jpg").convert("RGB")),
        instruction="pick up the cup",
    )
    import pdb; pdb.set_trace()