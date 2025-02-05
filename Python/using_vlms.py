"""
Module to test using VLMs (Vision Language Model)
"""
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image (replace 'path_to_image.jpg' with your image path)
image = Image.open("DataSetToTestGPTModels/hat_not_on_head.jpg")

# Define a list of text prompts (e.g., PPE compliance descriptions)
text_prompts = ["Worker wearing a hard hat on their head", "Worker not wearing a hard hat on their head"]

# Preprocess the image and text
inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

# Compute image-text similarity scores
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # Image-to-text similarity scores
probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

# Print the probabilities for each text prompt
for i, prompt in enumerate(text_prompts):
    print(f"Probability of '{prompt}': {probs[0][i].item():.4f}")
