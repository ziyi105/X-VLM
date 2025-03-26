from PIL import Image
import numpy as np
import torch
from transformers import BertTokenizer
from models.model_bbox import XVLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

# Load the model
yaml_config_path = "configs/Bbox_test.yaml"  # Path to your YAML file
with open(yaml_config_path, 'r') as f:
    config = yaml.safe_load(f)  # Load YAML as a Python dictionary

checkpoint_path = 'content/drive/MyDrive/fyp-dataset/checkpoints/checkpoint_best.pth'

model = XVLM(config=config)
model.load_state_dict(torch.load(checkpoint_path)['model'])
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the image
def preprocess_image(image_path, image_res):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_res, image_res))  # Resize to match model input
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    return image.unsqueeze(0)  # Add batch dimension

image_path = '/content/drive/MyDrive/fyp-dataset/test_images/cropped_scene_210.jpeg'  # Replace with your image path
image = preprocess_image(image_path, config['image_res']).to(device)

# Preprocess the text
text = "near the chair in front of the opened cabinet"  # Replace with your text description
text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)

# Pass the inputs to the model
with torch.no_grad():
    outputs_coord = model(image, text_input.input_ids, text_input.attention_mask, target_bbox=None)

# Get the predicted bounding box
pred_bbox = outputs_coord[0].cpu().numpy()  # Convert to numpy array
print("Predicted Bounding Box:", pred_bbox)

x_min, y_min, width, height = pred_bbox
x_max = x_min + width
y_max = y_min + height

# Load the original image
original_image = Image.open(image_path).convert('RGB')

# Plot the image and bounding box
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(original_image)

# Add the bounding box as a rectangle
rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Add a label for the bounding box
ax.text(x_min, y_min - 10, "Predicted BBox", color='red', fontsize=12, weight='bold')

plt.axis('off')  # Turn off the axis
plt.show()