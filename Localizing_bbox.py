import argparse
import os
import json
import random
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from models.model_bbox import XVLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb

os.environ["WANDB_API_KEY"] = "api_key" 

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, json_file, image_root, tokenizer, image_res, max_tokens):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_res = image_res
        self.max_tokens = max_tokens

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_res, self.image_res))  # Resize to match model input
        image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to CHW format
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.image_root, entry['image'])
        text = entry['text']
        bbox = entry['bbox']  # [x, y]

        # Preprocess image
        image = self.preprocess_image(image_path)

        # Tokenize text
        text_input = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_tokens,
            truncation=True,
            return_tensors="pt"
        )

        return image, text_input['input_ids'].squeeze(0), text_input['attention_mask'].squeeze(0), torch.tensor(bbox)

# Training Function
def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler):
    model.train()
    total_loss = 0

    for i, (images, input_ids, attention_mask, target_bbox) in enumerate(data_loader):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_bbox = target_bbox.to(device)

        # Forward pass
        _, loss_bbox, _ = model(images, input_ids, attention_mask, target_bbox=target_bbox)
        loss = loss_bbox

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    return total_loss / len(data_loader)

# Evaluation Function
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    results = []

    with torch.no_grad():
        for images, input_ids, attention_mask, target_bbox in data_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs_coord = model(images, input_ids, attention_mask, target_bbox=None)

            # Collect results
            for pred, target in zip(outputs_coord, target_bbox):
                results.append({
                    'pred_bbox': pred.cpu().numpy().tolist(),
                    'target_bbox': target.cpu().numpy().tolist()
                })

    return results

def main(args):
    # Initialize wandb
    wandb.init(
        project="localizing_bbox",  # Replace with your project name
        config={
            "train_file": args.train_file,
            "test_file": args.test_file,
            "image_root": args.image_root,
            "checkpoint": args.checkpoint,
            "output_dir": args.output_dir
        }
    )

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    # Create datasets and dataloaders
    train_dataset = CustomDataset(
        json_file=args.train_file,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_res=config['image_res'],
        max_tokens=config['max_tokens']
    )
    test_dataset = CustomDataset(
        json_file=args.test_file,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_res=config['image_res'],
        max_tokens=config['max_tokens']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, load_bbox_pretrain=True, is_eval=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Training loop
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, optimizer, tokenizer, epoch, device, scheduler)
        print(f"Epoch {epoch}, Training Loss: {train_loss}")

        # Log training loss to wandb
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # Evaluate
        results = evaluate(model, test_loader, tokenizer, device)
        print(f"Epoch {epoch}, Evaluation Results: {results}")

        # Log evaluation results to wandb
        for result in results:
            wandb.log({
                "epoch": epoch,
                "pred_bbox": result['pred_bbox'],
                "target_bbox": result['target_bbox']
            })

        # Save model checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Log checkpoint path to wandb
        wandb.log({"checkpoint_path": checkpoint_path})

    # Finalize wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help="Path to config file")
    parser.add_argument('--train_file', default='/content/drive/MyDrive/fyp-dataset/dataset/refcoco/train_dataset.json', type=str, required=False, help="Path to train JSON file")
    parser.add_argument('--test_file', default='/content/drive/MyDrive/fyp-dataset/dataset/refcoco/test/refcoco/test_dataset.json', type=str, required=False, help="Path to test JSON file")
    parser.add_argument('--image_root', default='/content/drive/MyDrive/fyp-dataset/test_images/', type=str, required=False, help="Path to image root directory")
    parser.add_argument('--checkpoint', default='/content/drive/MyDrive/fyp-dataset/checkpoints/checkpoint_best.pth', type=str, required=False, help="Path to pretrained checkpoint")
    parser.add_argument('--output_dir', default='output/localizing_bbox', type=str, required=False, help="Directory to save outputs")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)