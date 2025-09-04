1. BASIC BATCH STRUCTURE:
   
   Batch keys: ['sequence_length', 'sample_lens', 'packed_text_ids', 'packed_text_indexes', 'packed_position_ids', 'packed_vit_tokens', 'packed_vit_position_ids', 'packed_vit_token_indexes', 'vit_token_seqlens', 'padded_images', 'patchified_vae_latent_shapes', 'packed_vae_token_indexes', 'packed_timesteps', 'mse_loss_indexes', 'packed_label_ids', 'ce_loss_indexes', 'ce_loss_weights', 'nested_attention_masks', 'batch_data_indexes', 'num_tokens', 'data_indexes']
   Total sequence length: 2078

2. SEQUENCE STRUCTURE:
   Number of samples in batch: 1
   Sample lengths: [2078]
   Total tokens across samples: 2078
   Average sample length: 2078.0
   Min/Max sample length: 2078/2078

3. TEXT TOKENS:
   Text IDs shape: torch.Size([30])
   Text IDs dtype: torch.int64
   Vocab range: 6 - 151653
   Unique tokens: 30
   Most frequent tokens: [1, 1, 1, 1, 1]
   First 50 tokens decoded: '<|im_start|>The cover of a book titled 'Russell's Hardy Plants, 1930' with ornate designs and text.<|im_end|><|vision_start|><|vision_end|>...'
   Text indexes shape: torch.Size([30])
   Text index range: 0 - 2077
   Text positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 1052, 2077]...

   1 - 26 'real text tokens'


4. VIT TOKENS:
   ViT tokens shape: torch.Size([1024, 768])
   ViT tokens dtype: torch.float32
   Value range: [-1.0000, 1.0000]
   Mean: 0.2730, Std: 0.5136
   ViT token indexes shape: torch.Size([1024])
   ViT index range: 28 - 1051
   ViT positions: [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]...
   ViT position IDs shape: torch.Size([1024])
   ViT position range: 0 - 2201
   ViT sequence lengths: [1024]

5. VAE TOKENS:
   VAE images shape: torch.Size([1, 3, 512, 512])
   VAE images dtype: torch.float32
   Value range: [-1.0000, 1.0000]
   Mean: 0.2730, Std: 0.5136
   VAE token indexes shape: torch.Size([1024])
   VAE index range: 1053 - 2076
   VAE positions: [1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072]...
   VAE latent shapes: [[32, 32]]
   Total VAE tokens: 1024

6. ROPE POSITIONS:
   Position IDs shape: torch.Size([2078])
   Position range: 0 - 29
   Unique RoPE positions: 30
   RoPE positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
   Tokens per RoPE position: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1024), (29, 1026)]

7. ATTENTION MASKS:
   Number of attention masks: 1
   Mask 0 shape: torch.Size([2078, 2078])
2078



attention mask at vit: tensor([0., -inf, -inf,  ..., -inf, -inf, -inf])

attention mask at vae: tensor([0., 0., 0.,  ..., 0., 0., 0.])

8. LOSS COMPUTATION:
  
   CE loss range: 0 - 26
   Number of text tokens with loss: 27
   
   Weight range: 1.0000 - 1.0000
   Average weight: 1.0000
   Label IDs shape: torch.Size([27])
   Label range: 6 - 151645
   
   MSE loss indexes shape: torch.Size([1024])
   MSE loss range: 1053 - 2076
   Number of VAE tokens with loss: 1024
   Timesteps shape: torch.Size([1024])
   Valid timesteps: 1024
   Timestep range: 389.0 - 389.0
   Average timestep: 389.0

9. CONSISTENCY CHECKS:
   Declared sequence length: 2078
   Actual text IDs length: 30
   Length consistency: FAIL
   Index coverage: 2078 unique positions
   Index range: 0 - 2077
   Missing indexes: 0 (PASS)



15. BATCH SUMMARY:
   Text tokens: 30
   ViT tokens: 1024
   VAE tokens: 1024
   Total tokens: 2078
   Token distribution: 1.4% text, 49.3% ViT, 49.3% VAE
   
