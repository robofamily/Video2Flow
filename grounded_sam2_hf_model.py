import os
import copy
from pathlib import Path
import shutil
import tempfile
import cv2
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import supervision as sv
from supervision.draw.color import ColorPalette
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]
SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
OUTPUT_VIDEO_PATH = "./tracking_output.mp4"
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    # print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")

def crop_numpy_image(image, bbox):
    """
    Crop numpy image with bbox
    """
    x1, y1, x2, y2 = bbox

    # Avoid bound box out of image
    height, width = image.shape[:2]
    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))

    cropped_image = image[y1:y2, x1:x2]
    aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
    new_height = 256
    new_width = int(new_height * aspect_ratio)
    cropped_image = cv2.resize(cropped_image, (new_width, new_height))

    return cropped_image

class GroundedSAM:

    def __init__(
            self, 
            qwen,
            grounding_model_id,
            sam2_checkpoint,
            sam2_config,
            gd_api_token=None,
        ):

        self.device = torch.device("cuda")
        self.qwen = qwen
        if gd_api_token is None: # use local model
            self.use_gd_cloud = False
            self.gd_processor = AutoProcessor.from_pretrained(grounding_model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        elif gd_api_token:
            self.use_gd_cloud = True
            config = Config(gd_api_token)
            self.client = Client(config)
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.sam2_video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)

    def visualize(self, image, input_boxes, class_names, confidences, masks=None, fname=None):
        class_ids = np.array(list(range(len(class_names))))
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        if masks is not None:
            masks = masks.astype(bool)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks,  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        if fname is None:
            cv2.imwrite("./grounded_sam2_annotated_image_with_mask.jpg", annotated_frame)
        else:
            cv2.imwrite(fname, annotated_frame)

    def get_image_box(self, image, text, initial_threshold, visualize=False):
        """
        image: np.ndarray
        text: str
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if not self.use_gd_cloud:
                inputs = self.gd_processor(images=image, text=text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.grounding_model(**inputs)
                threshold = initial_threshold
                results = self.gd_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=[image.shape[:-1]]
                )
            elif self.use_gd_cloud:
                with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp_file:
                    image = Image.fromarray(image)
                    image.save(tmp_file, format="PNG")
                    tmp_file_path = tmp_file.name
                    url = self.client.upload_file(tmp_file_path)
                    task=DetectionTask(
                        image_url=url,
                        prompts=[TextPrompt(text=text)],
                        targets=[DetectionTarget.BBox],
                        model=DetectionModel.GDino1_6_Pro,
                    )
                    self.client.run_task(task)
                    result = task.result

        # Filter out low-confidence boxes
        if results[0]['scores'] == []:
            print("Text: " + text + " No object found")
            return None
        elif len(results[0]['scores']) > 1:
            new_results = [{"scores": [], "labels": [], "boxes": []}]
            for object_id in range(len(results[0]['scores'])):
                box = results[0]['boxes'][object_id].cpu().tolist()
                cropped_image = crop_numpy_image(image, box)
                good_detect = self.qwen.verify_object(cropped_image, results[0]['labels'][object_id])
                if good_detect:
                    new_results[0]['scores'].append(results[0]['scores'][object_id])
                    new_results[0]['labels'].append(results[0]['labels'][object_id])
                    new_results[0]['boxes'].append(results[0]['boxes'][object_id])
            if new_results[0]['scores'] == []:
                print("Text: " + text + " No object found")
                return None
            else:
                new_results[0]["scores"] = torch.stack(new_results[0]["scores"], dim=0)
                new_results[0]['boxes'] = torch.stack(new_results[0]['boxes'], dim=0)
                results = new_results

        if visualize:
            input_boxes = results[0]["boxes"].cpu().numpy()
            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
            self.visualize(image, input_boxes, class_names, confidences)

        return results

    def get_image_mask(self, image, text, initial_threshold, visualize=False):
        """
        image: np.ndarray
        text: str
        """

        results = self.get_image_box(image, text, initial_threshold, visualize=False)
        if results is None:
            return None, None
        input_boxes = results[0]["boxes"].cpu().numpy()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            self.sam2_predictor.set_image(image)
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            """
            Post-process the output of the model to get the masks, scores, and logits for visualization
            """
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            if visualize:
                confidences = results[0]["scores"].cpu().numpy().tolist()
                class_names = results[0]["labels"]
                self.visualize(image, input_boxes, class_names, confidences, masks)

        return masks, results

    def get_video_mask(self, video, text, initial_threshold, visualize=False):
        source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
        source_frames.mkdir(parents=True, exist_ok=True)
        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in video:
                sink.save_image(frame)

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # find all related object masks in the first frame
        masks, results = self.get_image_mask(
            image=video[0],
            text=text,
            initial_threshold=initial_threshold,
            visualize=False,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            # init video predictor state
            inference_state = self.sam2_video_predictor.init_state(
                video_path=SOURCE_VIDEO_FRAME_DIR,
            )

            # Add boxes to sam2 video tracker
            for object_id, box in enumerate(results[0]["boxes"], start=1):
                _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=object_id,
                    box=box.cpu().numpy(),
                )
            
            video_segments = {}  # video_segments contains the per-frame segmentation results

            # Track the objects according initial masks in frame 0
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_video_predictor.propagate_in_video(
                inference_state, 
                start_frame_idx=0,
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            if visualize:
                if os.path.exists(SAVE_TRACKING_RESULTS_DIR):
                    shutil.rmtree(SAVE_TRACKING_RESULTS_DIR)
                source_frames = Path(SAVE_TRACKING_RESULTS_DIR)
                source_frames.mkdir(parents=True, exist_ok=True)
                for frame_idx, segments in video_segments.items():
                    img = video[frame_idx]
                    masks = list(segments.values())
                    masks = np.concatenate(masks, axis=0)
                    confidences = results[0]["scores"].cpu().numpy().tolist()
                    class_names = results[0]["labels"]
                    fname = os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg")
                    self.visualize(img, sv.mask_to_xyxy(masks), class_names, confidences, masks, fname)
                create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)

        return video_segments
    
if __name__ == "__main__":
    gsam = GroundedSAM(
        grounding_model_id="IDEA-Research/grounding-dino-base",
        sam2_checkpoint="../Grounded-SAM-2/checkpoints/sam2_hiera_large.pt",
        sam2_config="sam2_hiera_l.yaml",
    )
    masks = gsam.get_image_mask(
        image=np.array(Image.open("../Grounded-SAM-2/notebooks/images/truck.jpg").convert("RGB")),
        text="car. tire.",
        initial_threshold=0.1,
        visualize=True,
    )