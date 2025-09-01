import torchvision.transforms as T
import webdataset as wds
from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
import torch

def load_config(path):
    return OmegaConf.load(path)


def WebDataset(shards, image_size=224, shuffle_buffer_size: int | None = None, split="train"):
    """
    Create a WebDataset for image-text pairs.
    """
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )

    if shuffle_buffer_size is None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shards),       # list of .tar files
            wds.split_by_worker,               # ensure workers get different shards
            wds.tarfile_to_samples(),          # stream tar contents
            wds.decode("pil"),                 # decode images with PIL
            wds.to_tuple("jpg", "txt"),        # read ("image", "caption") pairs
            wds.map(lambda x: {"image": transform(x[0]), "text": x[1]}),
        )
    else:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.shuffle(100),                  # shuffle shards
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle_buffer_size),  # shuffle samples
            wds.decode("pil"),
            wds.to_tuple("jpg", "txt"),
            wds.map(lambda x: {"image": transform(x[0]), "text": x[1]}),
        )

    return dataset

from torch.utils.data import DataLoader
import webdataset as wds
import yaml

def get_dataloader(split, cfg_path="/home/haoming/Bagel/config/test_data.yaml", dataset_cfg_path="/home/haoming/Bagel/data/configs/datacomp.yaml"):
    """Utility to instantiate a DataLoader from config."""
    cfg = load_config(cfg_path)
    dataset_cfg = load_config(dataset_cfg_path)

    # pick the right config block
    if split.lower() == "train":
        split_cfg = dataset_cfg.train
    elif split.lower() == "val":
        split_cfg = dataset_cfg.validation
    else:
        raise ValueError(f"Unknown split: {split}")

    dataset = WebDataset(split_cfg.shards, split_cfg.image_size)

    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 4),
    )

def save_batch_images(batch, output_dir="/home/haoming/Bagel/image"):
    """Save images from a batch to the output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    images = batch["image"]
    texts = batch["text"]
    batch_size = images.shape[0]
    
    print(f"Saving {batch_size} images to {output_dir}")
    
    for i in range(batch_size):
        # Get single image and text
        img_tensor = images[i]  # Shape: [3, H, W]
        text = texts[i]
        
        # Convert tensor to numpy for matplotlib (CHW -> HWC)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(text[:100] + "..." if len(text) > 100 else text, wrap=True, fontsize=10)
        
        # Save image
        output_path = os.path.join(output_dir, f"image_{i:03d}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Also save caption as text file
        caption_path = os.path.join(output_dir, f"caption_{i:03d}.txt")
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved: {output_path}")
    
    print(f"All {batch_size} images saved successfully!")

if __name__ == "__main__":
    print("=== Testing Validation Dataset - Saving First Batch ===")
    
    val_loader = get_dataloader(split="val")
    
    try:
        # Get first batch
        batch = next(iter(val_loader))
        
        print(f"First batch shape: {batch['image'].shape}")
        print(f"Batch size: {batch['image'].shape[0]}")
        print(f"Sample text: {batch['text'][0][:100]}...")
        
        # Save the batch images
        save_batch_images(batch, output_dir="/home/haoming/Bagel/image")
        
    except Exception as e:
        print(f"Error while processing: {e}")
        import traceback
        traceback.print_exc()