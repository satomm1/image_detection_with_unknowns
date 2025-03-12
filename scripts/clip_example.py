import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray

# Requirements for Mobile Clip
import torch
from PIL import Image
import mobileclip

# Other packages
import numpy as np


if __name__ == "__main__":

    # Set the device to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='checkpoints/mobileclip_s0.pt', device=device, root_dir="../src/mobileclip")
    tokenizer = mobileclip.get_tokenizer('mobileclip_s0', root_dir="../src/mobileclip")

    img = Image.open("reference_image.jpg").convert('RGB')
    
    # Crop image only to cone region
    # x1 = 45
    # y1 = 200
    # x2 = 255
    # y2 = 444

    # Crop image only to bottle region
    x1 = 389
    y1 = 240
    x2 = 444
    y2 = 409
    img = img.crop((x1, y1, x2, y2))

    # save image to file
    # img.save("cropped_cone.png")
    img.save("cropped_bottle.png")

    # Preprocess the image
    image = preprocess(img).unsqueeze(0)
    
    text = tokenizer(["a filing cabinet", "a cone", "a computer tower", "a recycle bin", "a helmet"])
    text = text.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image = image.to(device, dtype=torch.float16)
        image_features = model.encode_image(image)

        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_scores = (100.0 * image_features @ text_features.T)
        text_probs = text_scores.softmax(dim=-1)

    print("Scores:", text_scores.cpu().numpy())
    print("Label probs:", text_probs.cpu().numpy())
