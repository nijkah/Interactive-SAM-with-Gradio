import gradio as gr
import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry, SamPredictor

MODEL_DICT = dict(
    vit_h="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    vit_l="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    vit_b="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
)

def show_mask(mask: np.ndarray, image: np.ndarray, random_color: bool = False) -> np.ndarray:
    """Visualize a mask on top of an image.

    Args:
        mask (np.ndarray): A 2D array of shape (H, W).
        image (np.ndarray): A 3D array of shape (H, W, 3).
        random_color (bool): Whether to use a random color for the mask.
    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with the mask
        visualized on top of the image.
    """
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

    image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'), 0.3, 0)
    return image
    
def show_points(coords: np.ndarray, labels: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Visualize points on top of an image.
    
    Args:
        coords (np.ndarray): A 2D array of shape (N, 2).
        labels (np.ndarray): A 1D array of shape (N,).
        image (np.ndarray): A 3D array of shape (H, W, 3).
    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with the points
        visualized on top of the image.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    for p in pos_points:
        image = cv2.circle(image, p.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    for p in neg_points:
        image = cv2.circle(image, p.astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    return image
    
def setup_model() -> SamPredictor:
    """Setup the model and predictor.

    Returns:
        SamPredictor: The predictor.
    """

    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type]()
    sam.load_state_dict(torch.utils.model_zoo.load_url(MODEL_DICT[model_type]))
    sam.half()
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor

predictor = setup_model()

with gr.Blocks() as demo:

    # Define the UI
    mask_level = gr.Slider(minimum=0, maximum=2, value=1, step=1,\
                           label="Masking level",
                           info="(Whole - Part - Subpart) level")
    # with gr.Row():
    input_img = gr.Image(label="Input")
    output_img = gr.Image(label="Selected Segment")

    is_positive_box = gr.Checkbox(value=True, label='Positive point')
    reset = gr.Button("Reset Points")

    # Define the logic
    saved_points = []
    saved_labels = []

    def set_image(img) -> None:
        """Set the image for the predictor."""
        with torch.cuda.amp.autocast():
            predictor.set_image(img)

    def segment_anything(img, mask_level: int, is_positive: bool, evt: gr.SelectData):
        """Segment the selected region."""
        mask_level = 2 - mask_level
        saved_points.append([evt.index[0], evt.index[1]])
        saved_labels.append(1 if is_positive else 0)
        input_point = np.array(saved_points)
        input_label = np.array(saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        # mask has a shape of [3, h, w]
        masks = masks[mask_level:mask_level+1, ...]

        # Visualize the mask
        res = show_mask(masks, img)
        # Visualize the points
        res = show_points(input_point, input_label, res)
        return res
    
    def reset_points() -> None:
        """Reset the points."""
        global saved_points
        global saved_labels
        saved_points= []
        saved_labels = []

    # Connect the UI and logic
    input_img.upload(set_image, [input_img])
    input_img.select(segment_anything, [input_img, mask_level, is_positive_box], output_img)
    reset.click(reset_points)

if __name__ == "__main__":
    demo.launch()
