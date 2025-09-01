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
from data.data_utils import add_special_tokens



# Todos:
# 1. masking: manually assign special tokens casual masks or no? -- remove image start/iomage end added to vilex, include image start/end added to bagel
# 2. Loss computation: look into loss and make suree it worked
# 3. Get a test run: load this data, put into model, and see how it goes


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
        image_size=512,
        vae_image_downsample=16,
        vit_patch_size=16,
        max_sequence_length=4096,
        shuffle_buffer_size=1000,
    ):
        self.tokenizer = tokenizer
        self.vae_model = vae_model
        self.image_size = image_size
        self.vae_image_downsample = vae_image_downsample
        self.vit_patch_size = vit_patch_size
        self.max_sequence_length = max_sequence_length
        
        # Set special tokens as attributes
        for k, v in special_tokens.items():
            setattr(self, k, v)
            
        # Image transform
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
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
                            vae_h, vae_w, num_vit_patches, num_vae_tokens):
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
        
        # Add BOS + text tokens
        shifted_text = [self.bos_token_id] + text_tokens
        packed_text_ids.extend(shifted_text)
        packed_text_indexes.extend(range(curr, curr + len(shifted_text)))
        
        # Text loss
        ce_loss_indexes.extend(range(curr, curr + len(shifted_text)))
        ce_loss_weights.extend([1.0] * len(shifted_text))
        packed_label_ids.extend(text_tokens + [self.eos_token_id])
        
        curr += len(shifted_text)
        text_split_len += len(shifted_text)
        
        # Add EOS token
        packed_text_ids.append(self.eos_token_id)
        packed_text_indexes.append(curr)
        curr += 1
        text_split_len += 1
        
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
        packed_vit_token_indexes.extend(range(curr, curr + num_vit_patches))
        packed_vit_tokens.append(vit_patches)
        
        # ViT position IDs (2D flattened)
        h, w = image_tensor.shape[1:]
        from data.data_utils import get_flattened_position_ids_extrapolate
        vit_pos_ids = get_flattened_position_ids_extrapolate(
            h, w, self.vit_patch_size, max_num_patches_per_side=70
        )
        packed_vit_position_ids.append(vit_pos_ids)
        
        curr += num_vit_patches
        vit_split_len += num_vit_patches
        
        # End of image
        # packed_text_ids.append(self.end_of_image)

        # packed_text_indexes.append(curr)
        # curr += 1
        # vit_split_len += 1
        
        # All ViT tokens get same RoPE position

        
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
        timestep = np.random.randint(0, 1000)
        packed_timesteps.extend([timestep] * num_vae_tokens)
        
        curr += num_vae_tokens
        vae_split_len += num_vae_tokens
        
        # End of image
        packed_text_ids.append(self.end_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        vae_split_len += 1
        
        # All VAE tokens get same RoPE position
        packed_position_ids.extend([rope_id] * vae_split_len)
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
            'nested_attention_masks': [self._prepare_attention_mask(split_lens, attn_modes)],
            
            # Metadata
            'batch_data_indexes': [{'data_indexes': [0, 0, 0], 'worker_id': 0, 'dataset_name': 'simple'}],
            'num_tokens': curr,
            'data_indexes': {'data_indexes': [0, 0, 0], 'worker_id': 0, 'dataset_name': 'simple'},
        }
    
    def _prepare_attention_mask(self, split_lens, attn_modes):
        """Create attention mask for the sequence"""
        from data.data_utils import prepare_attention_mask_per_sample
        return prepare_attention_mask_per_sample(split_lens, attn_modes)
    
    def __iter__(self):
        for sample in self.dataset:
            if sample['num_tokens'] <= self.max_sequence_length:
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

def analyze_bagel_batch(batch, tokenizer=None, detailed=True):
    """
    Comprehensive analysis of a BAGEL batch with detailed statistics
    """
    
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
    
    # 9. SEQUENCE CONSISTENCY CHECKS
    print("\n9. CONSISTENCY CHECKS:")
    
    total_seq_len = batch.get('sequence_length', len(batch.get('packed_text_ids', [])))
    text_len = len(batch.get('packed_text_ids', []))
    
    print(f"   Declared sequence length: {total_seq_len}")
    print(f"   Actual text IDs length: {text_len}")
    print(f"   Length consistency: {'PASS' if total_seq_len == text_len else 'FAIL'}")
    
    # Check index ranges
    all_indexes = []
    if text_indexes is not None:
        all_indexes.extend(text_indexes.tolist())
    if vit_indexes is not None:
        all_indexes.extend(vit_indexes.tolist())
    if vae_indexes is not None:
        all_indexes.extend(vae_indexes.tolist())
    
    if all_indexes:
        all_indexes = sorted(set(all_indexes))
        expected_range = list(range(max(all_indexes) + 1))
        missing_indexes = set(expected_range) - set(all_indexes)
        
        print(f"   Index coverage: {len(all_indexes)} unique positions")
        print(f"   Index range: 0 - {max(all_indexes)}")
        print(f"   Missing indexes: {len(missing_indexes)} ({'PASS' if len(missing_indexes) == 0 else 'FAIL'})")
        if missing_indexes and len(missing_indexes) < 20:
            print(f"   Missing: {sorted(missing_indexes)}")
    
    # 10. MEMORY USAGE
    print("\n10. MEMORY USAGE:")
    
    total_elements = 0
    total_bytes = 0
    tensor_stats = []
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            elements = value.numel()
            bytes_used = elements * value.element_size()
            total_elements += elements
            total_bytes += bytes_used
            tensor_stats.append((key, elements, bytes_used))
    
    # Sort by memory usage
    tensor_stats.sort(key=lambda x: x[2], reverse=True)
    
    for key, elements, bytes_used in tensor_stats:
        print(f"   {key:25}: {elements:8,} elements, {bytes_used/1024/1024:6.2f} MB")
    
    print(f"   {'TOTAL':25}: {total_elements:8,} elements, {total_bytes/1024/1024:6.2f} MB")
    
    # 15. GENERATE SUMMARY REPORT
    print("\n15. BATCH SUMMARY:")
    
    total_text_tokens = len(text_indexes) if text_indexes is not None else 0
    total_vit_tokens = len(vit_indexes) if vit_indexes is not None else 0
    total_vae_tokens = len(vae_indexes) if vae_indexes is not None else 0
    
    print(f"   Text tokens: {total_text_tokens}")
    print(f"   ViT tokens: {total_vit_tokens}")
    print(f"   VAE tokens: {total_vae_tokens}")
    print(f"   Total tokens: {total_text_tokens + total_vit_tokens + total_vae_tokens}")
    
    if total_text_tokens + total_vit_tokens + total_vae_tokens > 0:
        text_pct = total_text_tokens / (total_text_tokens + total_vit_tokens + total_vae_tokens) * 100
        vit_pct = total_vit_tokens / (total_text_tokens + total_vit_tokens + total_vae_tokens) * 100  
        vae_pct = total_vae_tokens / (total_text_tokens + total_vit_tokens + total_vae_tokens) * 100
        
        print(f"   Token distribution: {text_pct:.1f}% text, {vit_pct:.1f}% ViT, {vae_pct:.1f}% VAE")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'total_tokens': total_text_tokens + total_vit_tokens + total_vae_tokens,
        'text_tokens': total_text_tokens,
        'vit_tokens': total_vit_tokens, 
        'vae_tokens': total_vae_tokens,
        'memory_mb': total_bytes / 1024 / 1024,
        'batch_keys': list(batch.keys()),
    }


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

    