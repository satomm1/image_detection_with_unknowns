import torch
from PIL import Image
import mobileclip
import time

# Set the device to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t1 = time.time()

# Load the model and tokenizer
model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='checkpoints/mobileclip_s0.pt', device=device)
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

t2 = time.time()

image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)

t3 = time.time()

text = tokenizer(["a diagram", "a dog", "a cat", "a cone"])
text = text.to(device)

t4 = time.time()


with torch.no_grad(), torch.cuda.amp.autocast():
    image = image.to(device, dtype=torch.float16)
    image_features = model.encode_image(image)

    t5 = time.time()

    text_features = model.encode_text(text)

    t6 = time.time()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_scores = (100.0 * image_features @ text_features.T)
    text_probs = text_scores.softmax(dim=-1)

print("Scores:", text_scores.cpu().numpy())
print("Label probs:", text_probs.cpu().numpy())

print("Time to load model:", t2 - t1)
print("Time to preprocess image:", t3 - t2)
print("Time to preprocess text:", t4 - t3)
print("Time to encode image:", t5 - t4)
print("Time to encode text:", t6 - t5)


image = preprocess(Image.open("docs/cone.jpg").convert('RGB')).unsqueeze(0)
with torch.no_grad(), torch.cuda.amp.autocast():
    t7 = time.time()
    image = image.to(device, dtype=torch.float16)
    image_features = model.encode_image(image)
    t8 = time.time()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_scores = (100.0 * image_features @ text_features.T)
    text_probs = text_scores.softmax(dim=-1)

print("Scores:", text_scores.cpu().numpy())
print("Label probs:", text_probs.cpu().numpy())
print("Time to encode image:", t8 - t7)


image = preprocess(Image.open("docs/chair.jpg").convert('RGB')).unsqueeze(0)
with torch.no_grad(), torch.cuda.amp.autocast():
    t9 = time.time()
    image = image.to(device, dtype=torch.float16)
    image_features = model.encode_image(image)
    t10 = time.time()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_scores = (100.0 * image_features @ text_features.T)
    text_probs = text_scores.softmax(dim=-1)

print("Scores:", text_scores.cpu().numpy())
print("Label probs:", text_probs.cpu().numpy())
print("Time to encode image:", t10 - t9)