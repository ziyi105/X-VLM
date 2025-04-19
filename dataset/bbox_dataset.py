import os
import json
import math
import random
from random import random as rand

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, resize
from PIL import Image

class bbox_dataset(Dataset):
    def __init__(self, json_path, transform=None, image_res=224, careful_hflip=False, mode='train'):
        """
        Args:
            json_path (str): Path to the JSON file containing dataset information.
            image_root (str): Root directory to prepend to image paths in the JSON file.
            transform (callable, optional): Transform to apply to the images.
            image_res (int): Resolution to resize images to.
            careful_hflip (bool): Whether to avoid horizontal flipping for captions with "left" or "right".
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_root = "/scratch/e0735461/x-vlm/dataset/refcoco/images"
        self.transform = transform
        self.image_res = image_res
        self.careful_hflip = careful_hflip
        self.mode = mode


    def __len__(self):
        return len(self.data)

    def left_or_right_in_caption(self, caption):
        """Check if the caption contains 'left' or 'right'."""
        return 'left' in caption or 'right' in caption

    def __getitem__(self, idx):
        item = self.data[idx]
        caption = item['text']

        # Load and preprocess the image
        image_path = os.path.join(self.image_root, item['image'])
        image = Image.open(image_path).convert('RGB')
        W, H = image.size

        if self.mode == 'train':
            # Extract bounding box
            x, y, w, h = item['bbox']
            assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (h > 0), "Invalid bounding box"

            # Random cropping
            x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
            x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H), H)
            w0, h0 = x1 - x0, y1 - y0
            assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (h0 > 0), "Invalid crop"
            image = image.crop((x0, y0, x0 + w0, y0 + h0))

            W, H = image.size

            # Horizontal flipping
            do_hflip = False
            if rand() < 0.5:
                if self.careful_hflip and self.left_or_right_in_caption(caption):
                    pass
                else:
                    image = hflip(image)
                    do_hflip = True

            # Resize the image
            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
            if self.transform:
                image = self.transform(image)

            # Adjust bounding box for cropping and flipping
            x = x - x0
            y = y - y0

            if do_hflip:  # Adjust for horizontal flipping
                x = (W - x) - w

            # Adjust bounding box for resizing
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            # Convert bounding box to center format and normalize
            center_x = x + 0.5 * w
            center_y = y + 0.5 * h
            target_bbox = torch.tensor([center_x / self.image_res, center_y / self.image_res,
                                        w / self.image_res, h / self.image_res], dtype=torch.float)

            return image, caption, target_bbox, item['image']
        
        else:
            image = self.transform(image)  # test_transform
            return image, caption, item['image']
    
    def get_ground_truth(self, image):
        """
        Retrieve the ground truth bounding box and image metadata for a given image_id.

        Args:
            image_id (str): The ID of the image.

        Returns:
            dict: A dictionary containing:
                - 'bbox': Ground truth bounding box [x, y, w, h].
                - 'width': Width of the image.
                - 'height': Height of the image.
        """
        # Find the annotation corresponding to the given image_id
        for item in self.data:
            if item['image'] == image:
                # Load the image to get its dimensions
                image_path = os.path.join(self.image_root, item['image'])
                with Image.open(image_path) as img:
                    width, height = img.size

                return {
                    'bbox': item['bbox'],  # [x, y, w, h]
                    'width': width,
                    'height': height
                }

        raise ValueError(f"Image {image} not found in the dataset.")