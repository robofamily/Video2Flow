import re
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
system_prompt = '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'
chat_template_config = ChatTemplateConfig('internvl-internlm2')
chat_template_config.meta_instruction = system_prompt
pipe = pipeline(model, chat_template_config=chat_template_config,
                backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
images = [load_image(img_url) for img_url in image_urls]
prompts = [('Please find the main target in this image and answer with a single word to describe the main target. For example. if an apple locates in the center of image, your answer should be: apple', img) for img in images]
responses = pipe(prompts)
targets = [response.text for response in responses]
print(targets)

prompts = [(f'Locate the <ref>{target}</ref> with bounding box in the image.', img) for (target, img) in zip(targets, images)]
responses = pipe(prompts)
bbox_texts = [response.text for response in responses]
bboxes = []
for bbox_text in bbox_texts:
    match = re.search(r'\[\[(\d+, \d+, \d+, \d+)\]\]', bbox_text)
    if match:
        array_str = match.group(1)
        bbox = [int(num) for num in array_str.split(', ')]
        print(bbox)
        bboxes.append(bbox)
    else:
        print("No match found")
import pdb; pdb.set_trace()
