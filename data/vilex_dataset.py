import torch
import torch.nn.functional as F
import torchvision.transforms as T
import webdataset as wds
import random
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from data.data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    add_special_tokens
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math


# Todos:
# 1. look at vilex dataloader and understand -- Done
# 2. Maintaining the same output structure, but use vilex dataloading + smaller dataset -- Done
# 3. Diagnose output attention and loss graphs
# 4/ 









class VilexDataset(torch.utils.data.IterableDataset):
    """
    Loads data and set basic configs
    #todo: make config load from yaml file so more convertable
    use create_webdataset to load samples, and handover to process sample ot do preprocessing
    """
    
    def __init__(
        self,
        shards,
        tokenizer,
        special_tokens,
        vae_model=None,
        max_image_size=512,  # Changed from image_size to max_image_size
        vae_image_downsample=64,
        vit_patch_size=14,
        max_sequence_length=4096,
        shuffle_buffer_size=1000,
    ):
        self.tokenizer = tokenizer
        self.vae_model = vae_model
        self.max_image_size = max_image_size
        self.vae_image_downsample = vae_image_downsample
        self.vit_patch_size = vit_patch_size
        self.max_sequence_length = max_sequence_length
        
        # Set special tokens as attributes
        for k, v in special_tokens.items():
            setattr(self, k, v)
            
        # Calculate actual image size (divisible by patch_size) -- originally just resizes to vae or vit, now resize to the lcm of both
        lcm_size = math.lcm(vit_patch_size, vae_image_downsample)
        self.actual_image_size = (max_image_size // lcm_size) * lcm_size
        print(f"Adjusted image size from {max_image_size} to {self.actual_image_size} (divisible by LCM({vit_patch_size}, {vae_image_downsample}) = {lcm_size})")
        
        # Image transform with dynamic sizing
        self.transform = T.Compose([
            T.Resize(self.actual_image_size),
            T.CenterCrop(self.actual_image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
        
        self.dataset = self._create_webdataset(shards, shuffle_buffer_size)
    
    def _create_webdataset(self, shards, shuffle_buffer_size):
        if shuffle_buffer_size > 0:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(shards),
                wds.shuffle(100),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(shuffle_buffer_size),
                wds.decode("pil"),
                wds.to_tuple("jpg", "txt"),
                wds.map(self._process_sample),
            )
        else:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(shards),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("pil"),
                wds.to_tuple("jpg", "txt"),
                wds.map(self._process_sample),
            )
        return dataset
    
    def _process_sample(self, sample):
        """Process single sample into BAGEL format with fixed sequence
        tokenize text, pachify vit, and keep vae the same
        pass the components and calculated num tokens into pack simple sequence for packing things together
        """
        image, text = sample
        
        # Preprocess
        image_tensor = self.transform(image)
        text_tokens = self.tokenizer.encode(text.strip())
        
        # Patchify for ViT
        from data.data_utils import patchify
        vit_patches = patchify(image_tensor, self.vit_patch_size)
        num_vit_patches = vit_patches.shape[0]
        
        # VAE dimensions  
        vae_h = image_tensor.shape[1] // self.vae_image_downsample
        vae_w = image_tensor.shape[2] // self.vae_image_downsample
        num_vae_tokens = vae_h * vae_w
        
        # FIXED SEQUENCE: text + vit + vae
        return self._pack_sequence(
            text_tokens, image_tensor, vit_patches, 
            vae_h, vae_w, num_vit_patches, num_vae_tokens
        )
    
    def _pack_sequence(self, text_tokens, image_tensor, vit_patches, 
                            vae_h, vae_w, num_vit_patches, num_vae_tokens,num_queries = 32):
        """Pack sequence in fixed order: text + vit + vae
        also includes meta data of the indexes (where these tokens appear in the sequence)
        length of sequence, position ids of different indexes, loss indexes (which part is got to compute which loss)
        and finally mse and ce loss weights
        #to do: look into controlling attn_modes and positional ids
        """
        
        # Initialize all arrays
        packed_text_ids = []
        packed_text_indexes = []
        packed_position_ids = []
        packed_vit_tokens = []
        packed_vit_position_ids = []
        packed_vit_token_indexes = []
        packed_vae_token_indexes = []
        packed_timesteps = []
        mse_loss_indexes = []
        packed_label_ids = []
        ce_loss_indexes = []
        ce_loss_weights = []
        
        curr = 0  # Absolute position tracker
        rope_id = 0  # RoPE position tracker
        split_lens = []
        attn_modes = []
        
        # 1. TEXT BLOCK
        text_split_len = 0
        
        # Add BOS + text tokens + Eos
        shifted_text = [self.bos_token_id] + text_tokens
        packed_text_ids.extend(shifted_text)
        packed_text_indexes.extend(range(curr, curr + len(shifted_text)))
        
        # Text loss
        ce_loss_indexes.extend(range(curr, curr + len(shifted_text)))
        ce_loss_weights.extend([len2weight(len(shifted_text))] * len(shifted_text))
        curr += len(shifted_text)
        text_split_len += len(shifted_text)
        packed_label_ids.extend([shifted_text])
        # Add EOS token -- special token (eos token) currently disabled loss but could potentially have loss

        
        
        # Text uses sequential RoPE positions
        packed_position_ids.extend(range(rope_id, rope_id + text_split_len))
        rope_id += text_split_len
        split_lens.append(text_split_len)
        attn_modes.append("causal")
        
        # 2. VIT BLOCK  
        vit_split_len = 0
        
        # Start of image -  not adding image tokens to vit for now
        # packed_text_ids.append(self.start_of_image)
        # packed_text_indexes.append(curr)
        # curr += 1
        # vit_split_len += 1
        
        # ViT tokens
        packed_vit_token_indexes.extend(range(curr, curr + num_queries))
        packed_vit_tokens.append(vit_patches)
        curr += num_queries
        vit_split_len += num_queries

        packed_vit_tokens.extend([self.eos_token_id])
        packed_vit_token_indexes.append(curr)
        curr += 1
        vit_split_len += 1
        

        # ViT position IDs (2D flattened)
        h, w = image_tensor.shape[1:]
        vit_pos_ids = get_flattened_position_ids_extrapolate(
            h, w, self.vit_patch_size, max_num_patches_per_side = 70
        )
        packed_vit_position_ids.append(vit_pos_ids)
        
        # End of image -- currently not applied
        # packed_text_ids.append(self.end_of_image)

        # packed_text_indexes.append(curr)
        # curr += 1
        # vit_split_len += 1
        
        # All ViT tokens get same RoPE position as text

        
        packed_position_ids.extend(range(rope_id, rope_id + vit_split_len))
        rope_id += vit_split_len
        split_lens.append(vit_split_len)
        attn_modes.append("causal")
        
        # 3. VAE BLOCK
        vae_split_len = 0
        
        # Start of image
        packed_text_ids.append(self.start_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        vae_split_len += 1
        
        # VAE tokens
        packed_vae_token_indexes.extend(range(curr, curr + num_vae_tokens))
        mse_loss_indexes.extend(range(curr, curr + num_vae_tokens))
        
        # Random timestep for diffusion
        timestep = np.random.randint(0, 1000) # no bound?
        packed_timesteps.extend([timestep] * num_vae_tokens)
        
        curr += num_vae_tokens
        vae_split_len += num_vae_tokens
        
        # End of image
        packed_text_ids.append(self.end_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        vae_split_len += 1
        
        # All VAE tokens get same RoPE position
        packed_latent_position_ids = []
        packed_position_ids.extend([rope_id] * vae_split_len)
        packed_latent_position_ids.append(
                    get_flattened_position_ids_extrapolate(
                        image_tensor.size(1), image_tensor.size(2),
                        self.vae_image_downsample, 
                        max_num_patches_per_side=70
                    )
                )
        rope_id += 1
        split_lens.append(vae_split_len)
        attn_modes.append("noise")  # For diffusion training
        
        # Convert to tensor format
        return {
            'sequence_length': curr,
            'sample_lens': [curr],  # Single sample
            'packed_text_ids': torch.tensor(packed_text_ids),
            'packed_text_indexes': torch.tensor(packed_text_indexes), 
            'packed_position_ids': torch.tensor(packed_position_ids),
            
            # ViT data
            'packed_vit_tokens': torch.cat([vit_patches], dim=0) if packed_vit_tokens else torch.empty(0),
            'packed_vit_position_ids': torch.cat(packed_vit_position_ids, dim=0) if packed_vit_position_ids else torch.empty(0),
            'packed_vit_token_indexes': torch.tensor(packed_vit_token_indexes),
            'vit_token_seqlens': torch.tensor([num_vit_patches]),
            'packed_latent_position_ids':torch.cat(packed_latent_position_ids, dim=0),
            
            # VAE data  
            'padded_images': image_tensor.unsqueeze(0),  # Add batch dim
            'patchified_vae_latent_shapes': [(vae_h, vae_w)],
            'packed_vae_token_indexes': torch.tensor(packed_vae_token_indexes),
            'packed_timesteps': torch.tensor(packed_timesteps),
            'mse_loss_indexes': torch.tensor(mse_loss_indexes),
            
            # Text loss data
            'packed_label_ids': torch.tensor(packed_label_ids),
            'ce_loss_indexes': torch.tensor(ce_loss_indexes), 
            'ce_loss_weights': torch.tensor(ce_loss_weights),
            
            # Attention
            "num_tokens": curr,
            'split_lens': split_lens,
            'attn_modes': attn_modes,
            'nested_attention_masks': [self._prepare_attention_mask(split_lens, attn_modes)],
            
            # Metadata
            'batch_data_indexes': [{'data_indexes': [0, 0, 0], 'worker_id': 0, 'dataset_name': 'simple'}],
            'data_indexes': {'data_indexes': [0, 0, 0], 'worker_id': 0, 'dataset_name': 'simple'},
        }
    
    def _prepare_attention_mask(self, split_lens, attn_modes):
        """Create attention mask for the sequence"""
        from data.data_utils import prepare_attention_mask_per_sample
        return prepare_attention_mask_per_sample(split_lens, attn_modes)
    
    def __iter__(self):
        for sample in self.dataset:
            if sample['num_tokens'] <= self.max_sequence_length:
                print("yielding sample with length" + str(sample["num_tokens"]))
                yield sample
                


def create_loader(
    shards,
    tokenizer, 
    special_tokens,
    vae_model=None,
    batch_size=1,
    num_workers=4,
    **kwargs
):
    """Create simplified BAGEL-compatible dataloader"""
    
    dataset = VilexDataset(
        shards=shards,
        tokenizer=tokenizer,
        special_tokens=special_tokens, 
        vae_model=vae_model,
        **kwargs
    )
    
    def collate_fn(batch):
        """Custom collate function for BAGEL compatibility"""
        # batch is a list with one item (since batch_size=None)
        sample = batch
        
        # Optionally wrap in BAGEL's SimpleCustomBatch for consistency
        # Or just return the sample directly
        return sample
    
    return DataLoader(
        dataset,
        batch_size=None,  # Already batched
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    from collections import Counter


def get_loader(split,tokenizer = None,special_tokens = None,vae_model = None, cfg_path = "/home/haoming/Bagel/data/configs/datacomp.yaml"):


    if tokenizer == None:
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    if special_tokens == None:
        special_tokens = new_token_ids

    if vae_model == None:
        vae_model, vae_config = load_ae("/home/haoming/Bagel/models/BAGEL-7B-MoT/ae.safetensors")
        vae_model.eval()

    dataset_cfg = OmegaConf.load(cfg_path)

    # Pick the right config block
    if split.lower() == "train":
        split_cfg = dataset_cfg.train
    elif split.lower() == "val":
        split_cfg = dataset_cfg.validation
    else:
        raise ValueError(f"Unknown split: {split}")
    
    return create_loader(shards = split_cfg.shards, tokenizer = tokenizer, special_tokens = special_tokens, vae_model=vae_model)


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_attention_mask(mask, text_len, vit_len, vae_len, title="Attention Mask", save_path=None):
    """
    Create a labeled heatmap of the attention mask
    """
    text_len -= 2
    vae_len +=2
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert mask to numpy for visualization (0 = white, -inf = black)
    mask_vis = mask.clone()
    mask_vis[mask_vis == float('-inf')] = -1
    mask_vis[mask_vis == 0] = 1
    
    # Create the heatmap
    im = ax.imshow(mask_vis.numpy(), cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
    
    # Add grid lines to separate sections
    text_end = text_len
    vit_end = text_len + vit_len
    vae_end = text_len + vit_len + vae_len
    
    # Vertical lines
    ax.axvline(x=text_end - 0.5, color='yellow', linewidth=2, alpha=0.8)
    ax.axvline(x=vit_end - 0.5, color='yellow', linewidth=2, alpha=0.8)
    
    # Horizontal lines  
    ax.axhline(y=text_end - 0.5, color='yellow', linewidth=2, alpha=0.8)
    ax.axhline(y=vit_end - 0.5, color='yellow', linewidth=2, alpha=0.8)
    
    # Add section labels
    ax.text(text_len//2, -2, 'TEXT', ha='center', fontweight='bold', fontsize=12)
    ax.text(text_len + vit_len//2, -2, 'VIT', ha='center', fontweight='bold', fontsize=12)
    ax.text(text_len + vit_len + vae_len//2, -2, 'VAE', ha='center', fontweight='bold', fontsize=12)
    
    ax.text(-2, text_len//2, 'TEXT', va='center', rotation=90, fontweight='bold', fontsize=12)
    ax.text(-2, text_len + vit_len//2, 'VIT', va='center', rotation=90, fontweight='bold', fontsize=12)
    ax.text(-2, text_len + vit_len + vae_len//2, 'VAE', va='center', rotation=90, fontweight='bold', fontsize=12)
    
    # Add tick labels for key positions
    major_ticks = [0, text_end-1, text_end, vit_end-1, vit_end, vae_end-1]
    major_labels = ['0', f'{text_end-1}', f'{text_end}', f'{vit_end-1}', f'{vit_end}', f'{vae_end-1}']
    
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(major_labels)
    
    # Add minor ticks for every 10th position if sequence is long
    if vae_end > 20:
        minor_ticks = list(range(0, vae_end, max(1, vae_end//10)))
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True, which='minor', alpha=0.3)
    
    ax.set_title(f'{title}\nWhite=Attend (0), Blue=Masked (-inf)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['-inf (Masked)', '0 (Attend)'])
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = 'attention_mask.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention mask saved to: {save_path}")
    
    plt.close()  # Close to free memory
    return save_path


def visualize_loss_sequence(sequence_length, ce_loss_indexes, mse_loss_indexes, 
                          text_len, vit_len, vae_len, title="Loss Index Sequence", save_path=None):
    """
    Create a sequence diagram showing which tokens have which type of loss
    """
    text_len -=2
    vae_len +=2
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Create base sequence
    sequence = list(range(sequence_length))
    colors = ['lightgray'] * sequence_length
    labels = ['No Loss'] * sequence_length
    
    # Mark CE loss positions (blue)
    if ce_loss_indexes is not None:
        for idx in ce_loss_indexes:
            if idx < sequence_length:
                colors[idx] = 'blue'
                labels[idx] = 'CE Loss'
    
    # Mark MSE loss positions (red) - may override CE
    if mse_loss_indexes is not None:
        for idx in mse_loss_indexes:
            if idx < sequence_length:
                colors[idx] = 'red'
                labels[idx] = 'MSE Loss'
    
    # Create bar chart
    bars = ax.bar(sequence, [1]*sequence_length, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add section dividers and labels
    text_end = text_len
    vit_end = text_len + vit_len
    vae_end = text_len + vit_len + vae_len
    
    # Vertical dividers
    ax.axvline(x=text_end - 0.5, color='yellow', linewidth=3, alpha=0.8)
    ax.axvline(x=vit_end - 0.5, color='yellow', linewidth=3, alpha=0.8)
    
    # Section background colors (subtle)
    ax.axvspan(-0.5, text_end - 0.5, alpha=0.1, color='green', label='Text Section')
    ax.axvspan(text_end - 0.5, vit_end - 0.5, alpha=0.1, color='orange', label='ViT Section')  
    ax.axvspan(vit_end - 0.5, vae_end - 0.5, alpha=0.1, color='purple', label='VAE Section')
    
    # Section labels at top
    ax.text(text_len//2, 1.1, 'TEXT TOKENS', ha='center', fontweight='bold', fontsize=12)
    ax.text(text_len + vit_len//2, 1.1, 'VIT TOKENS', ha='center', fontweight='bold', fontsize=12)
    ax.text(text_len + vit_len + vae_len//2, 1.1, 'VAE TOKENS', ha='center', fontweight='bold', fontsize=12)
    
    # Add position numbers for key locations
    key_positions = [0, text_end-1, text_end, vit_end-1, vit_end, vae_end-1]
    for pos in key_positions:
        if pos < sequence_length:
            ax.text(pos, -0.15, str(pos), ha='center', fontweight='bold', fontsize=10)
    
    # Add tick marks for every 10th position if sequence is long
    if sequence_length > 20:
        tick_positions = list(range(0, sequence_length, max(1, sequence_length//20)))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_positions], fontsize=8)
    else:
        ax.set_xticks(sequence)
        ax.set_xticklabels([str(i) for i in sequence], fontsize=8)
    
    ax.set_ylim(-0.2, 1.3)
    ax.set_xlim(-0.5, sequence_length - 0.5)
    ax.set_ylabel('Loss Type')
    ax.set_xlabel('Sequence Position')
    ax.set_title(f'{title}\nBlue=CE Loss, Red=MSE Loss, Gray=No Loss', fontsize=14, fontweight='bold')
    
    # Custom legend
    legend_elements = [
        patches.Patch(color='blue', alpha=0.7, label='CE Loss (Text)'),
        patches.Patch(color='red', alpha=0.7, label='MSE Loss (VAE)'),
        patches.Patch(color='lightgray', alpha=0.7, label='No Loss'),
        patches.Patch(color='green', alpha=0.1, label='Text Section'),
        patches.Patch(color='orange', alpha=0.1, label='ViT Section'),
        patches.Patch(color='purple', alpha=0.1, label='VAE Section')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = 'loss_sequence.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss sequence plot saved to: {save_path}")
    
    plt.close()  # Close to free memory
    return save_path


def analyze_bagel_batch(batch, tokenizer=None, detailed=True, save_dir="./plots"):
    """
    Comprehensive analysis of a BAGEL batch with detailed statistics and visualizations
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 80)
    print("BAGEL BATCH ANALYSIS")
    print("=" * 80)
    
    # ... existing analysis code ...
    # [Keep all the existing analysis code from lines 1-400+]
    
    # Get key tensor information for visualizations
    text_indexes = batch.get('packed_text_indexes')
    vit_indexes = batch.get('packed_vit_token_indexes')
    vae_indexes = batch.get('packed_vae_token_indexes')
    
    text_len = len(text_indexes) if text_indexes is not None else 0
    vit_len = len(vit_indexes) if vit_indexes is not None else 0
    vae_len = len(vae_indexes) if vae_indexes is not None else 0
    
    sequence_length = batch.get('sequence_length', text_len + vit_len + vae_len)
    
    # Loss indexes
    ce_loss_indexes = batch.get('ce_loss_indexes')
    mse_loss_indexes = batch.get('mse_loss_indexes')
    
    # [Keep all existing analysis code here]
    # ... (lines 1-500+ of existing analysis) ...
    
    # NEW: VISUALIZATION SECTION
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    saved_plots = []
    
    # 1. Attention Mask Visualization
    attn_masks = batch.get('nested_attention_masks')
    if attn_masks and len(attn_masks) > 0:
        print("Creating attention mask visualization...")
        
        mask = attn_masks[0]  # Take first mask
        attention_save_path = os.path.join(save_dir, 'attention_mask.png')
        saved_path = visualize_attention_mask(mask, text_len, vit_len, vae_len, 
                                             title="Attention Mask Structure",
                                             save_path=attention_save_path)
        saved_plots.append(saved_path)
        
        print(f"Attention mask shape: {mask.shape}")
        print(f"Sections - Text: 0-{text_len-1}, ViT: {text_len}-{text_len+vit_len-1}, VAE: {text_len+vit_len}-{text_len+vit_len+vae_len-1}")
    
    # 2. Loss Index Sequence Visualization
    print("Creating loss sequence visualization...")
    
    loss_save_path = os.path.join(save_dir, 'loss_sequence.png')
    saved_path = visualize_loss_sequence(sequence_length, ce_loss_indexes, mse_loss_indexes,
                                       text_len, vit_len, vae_len,
                                       title="Loss Computation Sequence",
                                       save_path=loss_save_path)
    saved_plots.append(saved_path)
    
    # Print loss statistics
    if ce_loss_indexes is not None:
        print(f"CE Loss positions: {ce_loss_indexes.tolist()}")
    if mse_loss_indexes is not None:
        print(f"MSE Loss positions: {mse_loss_indexes.tolist()}")
    
    print(f"Visualizations saved to: {saved_plots}")
    print("Visualizations complete!")
    
    print("=" * 80)
    print("BAGEL BATCH ANALYSIS")
    print("=" * 80)
    
    # 1. BASIC BATCH INFO
    print("\n1. BASIC BATCH STRUCTURE:")
    print(f"   Batch type: {type(batch)}")
    print(f"   Batch keys: {list(batch.keys())}")
    
    if hasattr(batch, 'sequence_length'):
        print(f"   Total sequence length: {batch.sequence_length}")
    elif 'sequence_length' in batch:
        print(f"   Total sequence length: {batch['sequence_length']}")
    
    # 2. SEQUENCE STRUCTURE ANALYSIS
    print("\n2. SEQUENCE STRUCTURE:")
    
    sample_lens = batch.get('sample_lens', batch.get('sample_lens', []))
    if sample_lens:
        print(f"   Number of samples in batch: {len(sample_lens)}")
        print(f"   Sample lengths: {sample_lens}")
        print(f"   Total tokens across samples: {sum(sample_lens)}")
        print(f"   Average sample length: {np.mean(sample_lens):.1f}")
        print(f"   Min/Max sample length: {min(sample_lens)}/{max(sample_lens)}")
    
    # 3. TEXT TOKEN ANALYSIS
    print("\n3. TEXT TOKENS:")
    
    text_ids = batch.get('packed_text_ids')
    if text_ids is not None:
        print(f"   Text IDs shape: {text_ids.shape}")
        print(f"   Text IDs dtype: {text_ids.dtype}")
        print(f"   Vocab range: {text_ids.min().item()} - {text_ids.max().item()}")
        
        # Token frequency analysis
        unique_tokens, counts = torch.unique(text_ids, return_counts=True)
        print(f"   Unique tokens: {len(unique_tokens)}")
        print(f"   Most frequent tokens: {torch.topk(counts, 5).values.tolist()}")
        
        if tokenizer:
            # Decode some tokens for inspection
            sample_tokens = text_ids[:50].tolist()
            decoded = tokenizer.decode(sample_tokens)
            print(f"   First 50 tokens decoded: '{decoded}...'")
    
    text_indexes = batch.get('packed_text_indexes')
    if text_indexes is not None:
        print(f"   Text indexes shape: {text_indexes.shape}")
        print(f"   Text index range: {text_indexes.min().item()} - {text_indexes.max().item()}")
        print(f"   Text positions: {text_indexes.tolist()}...")
    
    # 4. VIT TOKEN ANALYSIS  
    print("\n4. VIT TOKENS:")
    
    vit_tokens = batch.get('packed_vit_tokens')
    if vit_tokens is not None:
        print(f"   ViT tokens shape: {vit_tokens.shape}")
        print(f"   ViT tokens dtype: {vit_tokens.dtype}")
        print(f"   Value range: [{vit_tokens.min().item():.4f}, {vit_tokens.max().item():.4f}]")
        print(f"   Mean: {vit_tokens.mean().item():.4f}, Std: {vit_tokens.std().item():.4f}")
        
    
    vit_indexes = batch.get('packed_vit_token_indexes')
    if vit_indexes is not None:
        print(f"   ViT token indexes shape: {vit_indexes.shape}")
        print(f"   ViT index range: {vit_indexes.min().item()} - {vit_indexes.max().item()}")
        print(f"   ViT positions: {vit_indexes.tolist()[:20]}...")
    
    vit_pos_ids = batch.get('packed_vit_position_ids')
    if vit_pos_ids is not None:
        print(f"   ViT position IDs shape: {vit_pos_ids.shape}")
        print(f"   ViT position range: {vit_pos_ids.min().item()} - {vit_pos_ids.max().item()}")
        
    vit_seqlens = batch.get('vit_token_seqlens')
    if vit_seqlens is not None:
        print(f"   ViT sequence lengths: {vit_seqlens.tolist()}")
    
    # 5. VAE TOKEN ANALYSIS
    print("\n5. VAE TOKENS:")
    
    vae_images = batch.get('padded_images')
    if vae_images is not None:
        print(f"   VAE images shape: {vae_images.shape}")
        print(f"   VAE images dtype: {vae_images.dtype}")
        print(f"   Value range: [{vae_images.min().item():.4f}, {vae_images.max().item():.4f}]")
        print(f"   Mean: {vae_images.mean().item():.4f}, Std: {vae_images.std().item():.4f}")
    
    vae_indexes = batch.get('packed_vae_token_indexes') 
    if vae_indexes is not None:
        print(f"   VAE token indexes shape: {vae_indexes.shape}")
        print(f"   VAE index range: {vae_indexes.min().item()} - {vae_indexes.max().item()}")
        print(f"   VAE positions: {vae_indexes.tolist()[:20]}...")
    
    vae_latent_shapes = batch.get('patchified_vae_latent_shapes')
    if vae_latent_shapes:
        print(f"   VAE latent shapes: {vae_latent_shapes}")
        total_latent_tokens = sum(h * w for h, w in vae_latent_shapes)
        print(f"   Total VAE tokens: {total_latent_tokens}")
    
    vae_pos_ids = batch.get('packed_latent_position_ids')
    if vae_pos_ids is not None:
        print(f"   VAE position IDs shape: {vae_pos_ids.shape}")
        print(f"   VAE position range: {vae_pos_ids.min().item()} - {vae_pos_ids.max().item()}")
    
    # 6. POSITION AND ROPE ANALYSIS
    print("\n6. ROPE POSITIONS:")
    
    pos_ids = batch.get('packed_position_ids')
    if pos_ids is not None:
        print(f"   Position IDs shape: {pos_ids.shape}")
        print(f"   Position range: {pos_ids.min().item()} - {pos_ids.max().item()}")
        
        # Analyze RoPE pattern
        unique_positions = torch.unique(pos_ids)
        print(f"   Unique RoPE positions: {len(unique_positions)}")
        print(f"   RoPE positions: {unique_positions.tolist()}")
        
        # Count tokens per RoPE position
        rope_counts = [(pos.item(), (pos_ids == pos).sum().item()) for pos in unique_positions]
        print(f"   Tokens per RoPE position: {rope_counts}")
    
    # 7. ATTENTION MASK ANALYSIS
    print("\n7. ATTENTION MASKS:")
    
    attn_masks = batch.get('nested_attention_masks')
    if attn_masks:
        print(f"   Number of attention masks: {len(attn_masks)}")
        for i, mask in enumerate(attn_masks):
            print(f"   Mask {i} shape: {mask.shape}")



            text_len = text_indexes.shape[0]
            vit_len = vit_indexes.shape[0]
            vae_len = vae_indexes.shape[0]

            total = text_len + vit_len + vae_len
            print(batch["sequence_length"])
            print(text_len)
            print(vit_len)
            print(vae_len)

            print(mask[text_len + vit_len-3][text_len + vit_len - 3:])

            print(mask[-1][-3:])
            print(mask[text_len + vit_len + 1][text_len + vit_len:])
    
    # 8. LOSS COMPUTATION ANALYSIS
    print("\n8. LOSS COMPUTATION:")
    
    # Text loss
    ce_indexes = batch.get('ce_loss_indexes')
    ce_weights = batch.get('ce_loss_weights')
    label_ids = batch.get('packed_label_ids')
    
    if ce_indexes is not None:
        print(f"   CE loss indexes shape: {ce_indexes.shape}")
        print(f"   CE loss range: {ce_indexes.min().item()} - {ce_indexes.max().item()}")
        print(f"   Number of text tokens with loss: {len(ce_indexes)}")
    
    if ce_weights is not None:
        print(f"   CE loss weights shape: {ce_weights.shape}")
        print(f"   Weight range: {ce_weights.min().item():.4f} - {ce_weights.max().item():.4f}")
        print(f"   Average weight: {ce_weights.mean().item():.4f}")
    
    if label_ids is not None:
        print(f"   Label IDs shape: {label_ids.shape}")
        print(f"   Label range: {label_ids.min().item()} - {label_ids.max().item()}")
    
    # VAE loss
    mse_indexes = batch.get('mse_loss_indexes')
    timesteps = batch.get('packed_timesteps')
    
    if mse_indexes is not None:
        print(f"   MSE loss indexes shape: {mse_indexes.shape}")
        print(f"   MSE loss range: {mse_indexes.min().item()} - {mse_indexes.max().item()}")
        print(f"   Number of VAE tokens with loss: {len(mse_indexes)}")
    
    if timesteps is not None:
        print(f"   Timesteps shape: {timesteps.shape}")
        valid_timesteps = timesteps[timesteps != float('-inf')]
        if len(valid_timesteps) > 0:
            print(f"   Valid timesteps: {len(valid_timesteps)}")
            print(f"   Timestep range: {valid_timesteps.min().item():.1f} - {valid_timesteps.max().item():.1f}")
            print(f"   Average timestep: {valid_timesteps.float().mean().item():.1f}")
        else:
            print("   No valid timesteps found")
    return None


def test_dataloader_statistics(dataloader, tokenizer=None, num_batches=3):
    """
    Test multiple batches and provide aggregate statistics
    """
    
    print("=" * 80)
    print("MULTI-BATCH DATALOADER TESTING")
    print("=" * 80)
    
    batch_stats = []
    
    try:
        for i, batch in enumerate(dataloader):
            print(f"\n{'='*20} BATCH {i+1} {'='*20}")
            
            # Analyze this batch
            stats = analyze_bagel_batch(batch, tokenizer, detailed=(i==0))
            batch_stats.append(stats)
            
            if i >= num_batches - 1:
                break
                
    except Exception as e:
        print(f"Error during batch iteration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Aggregate statistics
    if batch_stats:
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)
        
        total_tokens = [s['total_tokens'] for s in batch_stats]
        text_tokens = [s['text_tokens'] for s in batch_stats]
        vit_tokens = [s['vit_tokens'] for s in batch_stats]
        vae_tokens = [s['vae_tokens'] for s in batch_stats]
        memory_usage = [s['memory_mb'] for s in batch_stats]
        
        print(f"Batches analyzed: {len(batch_stats)}")
        print(f"\nToken statistics across batches:")
        print(f"   Total tokens - Mean: {np.mean(total_tokens):.1f}, Std: {np.std(total_tokens):.1f}")
        print(f"   Text tokens - Mean: {np.mean(text_tokens):.1f}, Std: {np.std(text_tokens):.1f}")
        print(f"   ViT tokens - Mean: {np.mean(vit_tokens):.1f}, Std: {np.std(vit_tokens):.1f}")
        print(f"   VAE tokens - Mean: {np.mean(vae_tokens):.1f}, Std: {np.std(vae_tokens):.1f}")
        print(f"   Memory usage - Mean: {np.mean(memory_usage):.1f} MB, Std: {np.std(memory_usage):.1f} MB")
        
        print(f"\nToken ranges:")
        print(f"   Total: {min(total_tokens)} - {max(total_tokens)}")
        print(f"   Text: {min(text_tokens)} - {max(text_tokens)}")
        print(f"   ViT: {min(vit_tokens)} - {max(vit_tokens)}")
        print(f"   VAE: {min(vae_tokens)} - {max(vae_tokens)}")


# USAGE EXAMPLE:
if __name__ == "__main__":

    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    # Uncomment below to enable VAE encoding:
    vae_model, vae_config = load_ae("/home/haoming/Bagel/models/BAGEL-7B-MoT/ae.safetensors")
    vae_model.eval()

    train_loader = get_loader(split = "val")
    print("Testing single batch:")
    batch = next(iter(train_loader))
    analyze_bagel_batch(batch, tokenizer, save_dir="./bagel_analysis_plots")