import numpy as np
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont

def preprocess_image(image, detection=False):
    """Preprocesses the image for the model."""
    if detection:
        img_tensor = F.to_tensor(image).unsqueeze(0)
        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        return img_tensor
    else:
        # Preprocessing for ResNet18
        img = image.resize((256, 256))
        img = img.crop((16, 16, 240, 240))
        img = np.array(img)
        img = img / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)
        # Move to GPU if available
        if torch.cuda.is_available():
            img = img.cuda()
        return img

def draw_bounding_boxes(image, predictions, coco_names, confidence_threshold):
    """Draws bounding boxes on the image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i in range(len(predictions[0]["boxes"])):
        confidence = predictions[0]["scores"][i].item()
        if confidence > confidence_threshold:
            box = predictions[0]["boxes"][i].detach().numpy()
            label_index = predictions[0]["labels"][i].item()
            label = f"{coco_names[label_index]}: {confidence:.2f}"
            
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1] - 20), label, fill="red", font=font)
            
    return img_draw
