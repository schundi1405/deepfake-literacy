
from datasets import load_dataset
from pathlib import Path

# Load the cached dataset
dataset = load_dataset(
    "prithivMLmods/AI-vs-Deepfake-vs-Real",
    split="train"
)

# Create image folder
output_dir = Path("data/sample_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Select three examples from each category
fake_images = dataset.filter(lambda x: x["label"] == 1)
real_images = dataset.filter(lambda x: x["label"] == 2)

# Save images
for label, images in [("fake", fake_images), ("real", real_images)]:

    for i in range(3):
        image = images[i]["image"].convert("RGB")

        filename = f"{label}_{i + 1}.jpg"
        image.save(output_dir / filename)

        print(f"Saved {filename}")

print("\nFinished loading sample images!")
