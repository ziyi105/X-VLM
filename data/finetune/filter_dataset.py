import json

def validate_and_clean_dataset(input_path, output_path, W, H):
    with open(input_path, 'r') as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        if "bbox" not in item or not item["bbox"]:
            print(f"Skipping entry with missing or empty bbox: {item}")
            continue

        x, y, w, h = item["bbox"]
        # Validate bounding box
        if (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (h > 0):
            cleaned_data.append(item)
        else:
            print(f"Invalid bounding box: {item}")

    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

# Example usage
validate_and_clean_dataset(
    input_path="data/finetune/dataset.json",
    output_path="cleaned_dataset.json",
    W=224,  # Set your image width
    H=224   # Set your image height
)