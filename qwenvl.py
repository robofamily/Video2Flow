import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenObjectProposer:

    def __init__(self, model_id, device):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def infer(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(image), 
                    },
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
        prompt = "The instruction to the robot is: " \
        + instruction + \
        ". Please answer which object will be manipulated with a single word. " + \
        "For example, if the robotics arm will manipulate an apple, the answer should be: apple."
        object = self.infer(image, prompt)[0] + '.'
        return object
    
    def verify_object(self, image, object):
        prompt = "Is is a figure of " + object + "? Please answer with a single word. " + \
        "For example, if I ask you whether it is a figure of apple, and the main object of the figure is an apple, " + \
        "your answer should be: yes." + \
        "If I ask you whether it is a figure of apple, and the main object of the figure is a banana, " + \
        "your answer should be: no.",
        answer = self.infer(image, prompt)[0]
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