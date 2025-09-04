import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from ..vilex_dataset import VilexDataset

class TestPackSequence:
    """Test suite for VilexDataset._pack_sequence method"""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock VilexDataset instance for testing"""
        # Mock the required components
        mock_tokenizer = Mock()
        mock_special_tokens = {
            'bos_token_id': 1,
            'eos_token_id': 2, 
            'start_of_image': 3,
            'end_of_image': 4
        }
        
        # Create dataset instance
        dataset = VilexDataset(
            shards=[],
            tokenizer=mock_tokenizer,
            special_tokens=mock_special_tokens,
            vae_model=None,
            image_size=512,
            vae_image_downsample=16,
            vit_patch_size=16,
            max_sequence_length=4096,
            shuffle_buffer_size=0
        )
        
        return dataset

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for _pack_sequence"""
        # Sample text tokens
        text_tokens = [100, 101, 102, 103, 104]
        
        # Sample image tensor (3, 512, 512)
        image_tensor = torch.randn(3, 512, 512)
        
        # Sample ViT patches (1024, 768) - 32x32 patches of 768 dims
        vit_patches = torch.randn(1024, 768)
        
        # VAE dimensions
        vae_h, vae_w = 32, 32  # 512/16 = 32
        num_vit_patches = 1024
        num_vae_tokens = vae_h * vae_w  # 1024
        
        return {
            'text_tokens': text_tokens,
            'image_tensor': image_tensor,
            'vit_patches': vit_patches,
            'vae_h': vae_h,
            'vae_w': vae_w,
            'num_vit_patches': num_vit_patches,
            'num_vae_tokens': num_vae_tokens
        }

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_basic_structure(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test basic structure and keys of packed sequence output"""
        # Mock the position ID function to return proper tensors
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)  # Approximate size
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Check all required keys are present
        expected_keys = {
            'sequence_length', 'sample_lens', 'packed_text_ids', 'packed_text_indexes',
            'packed_position_ids', 'packed_vit_tokens', 'packed_vit_position_ids',
            'packed_vit_token_indexes', 'vit_token_seqlens', 'packed_latent_position_ids',
            'padded_images', 'patchified_vae_latent_shapes', 'packed_vae_token_indexes',
            'packed_timesteps', 'mse_loss_indexes', 'packed_label_ids', 'ce_loss_indexes',
            'ce_loss_weights', 'split_lens', 'attn_modes', 'nested_attention_masks',
            'batch_data_indexes', 'num_tokens', 'data_indexes'
        }
        
        assert set(result.keys()) == expected_keys
        assert isinstance(result, dict)

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_tensor_types(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test that all tensor outputs have correct types"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Check tensor types
        tensor_fields = [
            'packed_text_ids', 'packed_text_indexes', 'packed_position_ids',
            'packed_vit_tokens', 'packed_vit_position_ids', 'packed_vit_token_indexes',
            'vit_token_seqlens', 'packed_latent_position_ids', 'padded_images',
            'packed_vae_token_indexes', 'packed_timesteps', 'mse_loss_indexes',
            'packed_label_ids', 'ce_loss_indexes', 'ce_loss_weights'
        ]
        
        for field in tensor_fields:
            assert isinstance(result[field], torch.Tensor), f"{field} should be a tensor"

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample') 
    def test_pack_sequence_sequence_length_consistency(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test that sequence length is consistent across different components"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        sequence_length = result['sequence_length']
        
        # Text tokens: BOS + text + EOS = 1 + 5 + 1 = 7
        expected_text_len = 1 + len(sample_inputs['text_tokens']) + 1
        
        # ViT tokens: num_vit_patches = 1024  
        expected_vit_len = sample_inputs['num_vit_patches']
        
        # VAE tokens: start_of_image + vae_tokens + end_of_image = 1 + 1024 + 1 = 1026
        expected_vae_len = 1 + sample_inputs['num_vae_tokens'] + 1
        
        expected_total = expected_text_len + expected_vit_len + expected_vae_len
        
        assert sequence_length == expected_total
        assert len(result['packed_text_ids']) == sequence_length
        assert len(result['packed_position_ids']) == sequence_length

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_text_block(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test text block packing specifically"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Check text IDs structure: [BOS, ...text_tokens..., EOS]
        text_ids = result['packed_text_ids']
        expected_text_start = 7  # BOS + 5 text tokens + EOS
        
        assert text_ids[0].item() == mock_dataset.bos_token_id
        assert text_ids[expected_text_start - 1].item() == mock_dataset.eos_token_id
        
        # Check text indexes are sequential from 0
        text_indexes = result['packed_text_indexes']
        assert text_indexes[0].item() == 0
        assert text_indexes[-1].item() == expected_text_start - 1
        
        # Check CE loss configuration
        ce_indexes = result['ce_loss_indexes']
        ce_weights = result['ce_loss_weights'] 
        label_ids = result['packed_label_ids']
        
        assert len(ce_indexes) == expected_text_start
        assert len(ce_weights) == expected_text_start
        assert len(label_ids) == expected_text_start
        assert all(w == 1.0 for w in ce_weights)

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_vit_block(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test ViT block packing specifically"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # ViT tokens should match input
        vit_tokens = result['packed_vit_tokens']
        assert torch.equal(vit_tokens, sample_inputs['vit_patches'])
        
        # ViT indexes should start after text block
        text_len = 7  # BOS + 5 text + EOS
        vit_indexes = result['packed_vit_token_indexes']
        expected_vit_start = text_len
        expected_vit_end = text_len + sample_inputs['num_vit_patches']
        
        assert vit_indexes[0].item() == expected_vit_start
        assert vit_indexes[-1].item() == expected_vit_end - 1
        
        # Check ViT sequence lengths
        vit_seqlens = result['vit_token_seqlens']
        assert vit_seqlens[0].item() == sample_inputs['num_vit_patches']

    @patch('data.data_utils.get_flattened_position_ids_extrapolate') 
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_vae_block(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test VAE block packing specifically"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # VAE indexes should start after text + vit blocks
        text_len = 7
        vit_len = sample_inputs['num_vit_patches']
        vae_indexes = result['packed_vae_token_indexes']
        
        expected_vae_start = text_len + vit_len + 1  # +1 for start_of_image
        expected_vae_end = expected_vae_start + sample_inputs['num_vae_tokens']
        
        assert vae_indexes[0].item() == expected_vae_start
        assert vae_indexes[-1].item() == expected_vae_end - 1
        
        # Check VAE latent shapes
        vae_shapes = result['patchified_vae_latent_shapes']
        assert vae_shapes == [(sample_inputs['vae_h'], sample_inputs['vae_w'])]
        
        # Check MSE loss indexes match VAE indexes
        mse_indexes = result['mse_loss_indexes']
        assert torch.equal(mse_indexes, vae_indexes)
        
        # Check timesteps
        timesteps = result['packed_timesteps']
        assert len(timesteps) == sample_inputs['num_vae_tokens']
        assert all(0 <= t < 1000 for t in timesteps)

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_position_ids(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test RoPE position ID assignment"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        position_ids = result['packed_position_ids']
        
        # Text block should have sequential RoPE positions 0,1,2,3,4,5,6
        text_len = 7
        text_positions = position_ids[:text_len]
        expected_text_positions = torch.arange(text_len)
        assert torch.equal(text_positions, expected_text_positions)
        
        # ViT block should have sequential positions starting from text_len
        vit_len = sample_inputs['num_vit_patches']
        vit_positions = position_ids[text_len:text_len + vit_len]
        expected_vit_positions = torch.arange(text_len, text_len + vit_len)
        assert torch.equal(vit_positions, expected_vit_positions)
        
        # VAE block should all have the same RoPE position
        vae_start = text_len + vit_len
        vae_len = 1 + sample_inputs['num_vae_tokens'] + 1  # start + tokens + end
        vae_positions = position_ids[vae_start:vae_start + vae_len]
        expected_vae_position = text_len + vit_len
        assert all(pos == expected_vae_position for pos in vae_positions)

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_attention_modes(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test attention modes assignment"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        attn_modes = result['attn_modes']
        split_lens = result['split_lens']
        
        # Should have 3 splits: text, vit, vae
        assert len(attn_modes) == 3
        assert len(split_lens) == 3
        
        # Modes should be: causal, causal, noise
        assert attn_modes == ["causal", "causal", "noise"]
        
        # Split lengths should match expected
        expected_text_len = 7
        expected_vit_len = sample_inputs['num_vit_patches']
        expected_vae_len = 1 + sample_inputs['num_vae_tokens'] + 1
        
        assert split_lens[0] == expected_text_len
        assert split_lens[1] == expected_vit_len
        assert split_lens[2] == expected_vae_len

    def test_pack_sequence_empty_vit_tokens(self, mock_dataset):
        """Test handling of empty ViT tokens"""
        inputs = {
            'text_tokens': [100, 101],
            'image_tensor': torch.randn(3, 512, 512),
            'vit_patches': torch.empty(0, 768),  # Empty ViT patches
            'vae_h': 32,
            'vae_w': 32,
            'num_vit_patches': 0,
            'num_vae_tokens': 1024
        }
        
        with patch('data.data_utils.get_flattened_position_ids_extrapolate') as mock_pos_ids, \
             patch('data.data_utils.prepare_attention_mask_per_sample') as mock_attn_mask:
            
            mock_pos_ids.return_value = torch.arange(1024).reshape(-1, 1)
            mock_attn_mask.return_value = torch.eye(1030)
            
            result = mock_dataset._pack_sequence(**inputs)
            
            # Should handle empty ViT tokens gracefully
            assert result['packed_vit_tokens'].shape[0] == 0
            assert len(result['packed_vit_token_indexes']) == 0
            assert result['vit_token_seqlens'][0].item() == 0

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    def test_pack_sequence_latent_position_ids_concatenation_error(self, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test the specific error case that's causing the RuntimeError"""
        # This is the root cause - mock returns scalar instead of proper tensor
        mock_get_position_ids.return_value = torch.tensor(42)  # Zero-dimensional tensor
        
        with patch('data.data_utils.prepare_attention_mask_per_sample') as mock_attn_mask:
            mock_attn_mask.return_value = torch.eye(2050)
            
            # This should raise the RuntimeError we're seeing
            with pytest.raises(RuntimeError, match="zero-dimensional tensor.*cannot be concatenated"):
                mock_dataset._pack_sequence(**sample_inputs)

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_latent_position_ids_fix(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test that proper tensor shape fixes the concatenation issue"""
        # Return proper 2D tensor instead of scalar
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        # This should work without error
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Verify packed_latent_position_ids is properly formed
        latent_pos_ids = result['packed_latent_position_ids']
        assert isinstance(latent_pos_ids, torch.Tensor)
        assert latent_pos_ids.dim() >= 1  # Not zero-dimensional

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_metadata(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test metadata fields are properly set"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Check metadata
        assert result['sample_lens'] == [result['sequence_length']]
        assert result['num_tokens'] == result['sequence_length']
        
        # Check batch data indexes
        batch_data_indexes = result['batch_data_indexes']
        assert len(batch_data_indexes) == 1
        assert batch_data_indexes[0]['dataset_name'] == 'simple'
        assert batch_data_indexes[0]['worker_id'] == 0
        
        # Check data indexes
        data_indexes = result['data_indexes']
        assert data_indexes['dataset_name'] == 'simple'
        assert data_indexes['worker_id'] == 0

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_image_shapes(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test image tensor processing"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Check padded images
        padded_images = result['padded_images']
        assert padded_images.shape == (1, 3, 512, 512)  # Batch dim added
        assert torch.equal(padded_images[0], sample_inputs['image_tensor'])

    def test_pack_sequence_with_different_vae_dimensions(self, mock_dataset):
        """Test with different VAE dimensions"""
        inputs = {
            'text_tokens': [100, 101, 102],
            'image_tensor': torch.randn(3, 256, 256),  # Smaller image
            'vit_patches': torch.randn(256, 768),  # Fewer patches
            'vae_h': 16,  # 256/16 = 16
            'vae_w': 16,
            'num_vit_patches': 256,
            'num_vae_tokens': 256  # 16*16
        }
        
        with patch('data.data_utils.get_flattened_position_ids_extrapolate') as mock_pos_ids, \
             patch('data.data_utils.prepare_attention_mask_per_sample') as mock_attn_mask:
            
            mock_pos_ids.return_value = torch.arange(256).reshape(-1, 1)
            mock_attn_mask.return_value = torch.eye(520)  # Adjust size
            
            result = mock_dataset._pack_sequence(**inputs)
            
            # Check VAE-specific results
            assert result['patchified_vae_latent_shapes'] == [(16, 16)]
            assert len(result['packed_vae_token_indexes']) == 256
            assert len(result['mse_loss_indexes']) == 256

    @patch('numpy.random.randint')
    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_timestep_generation(self, mock_prepare_attention_mask, mock_get_position_ids, mock_randint, mock_dataset, sample_inputs):
        """Test timestep generation for diffusion"""
        # Mock random timestep
        mock_randint.return_value = 542
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Check timestep generation
        timesteps = result['packed_timesteps']
        assert all(t == 542 for t in timesteps)
        assert len(timesteps) == sample_inputs['num_vae_tokens']
        
        # Verify randint was called correctly
        mock_randint.assert_called_once_with(0, 1000)

    @patch('data.data_utils.get_flattened_position_ids_extrapolate')
    @patch('data.data_utils.prepare_attention_mask_per_sample')
    def test_pack_sequence_index_consistency(self, mock_prepare_attention_mask, mock_get_position_ids, mock_dataset, sample_inputs):
        """Test that all indexes are consistent and cover the full sequence"""
        mock_get_position_ids.return_value = torch.arange(1024).reshape(-1, 1)
        mock_prepare_attention_mask.return_value = torch.eye(2050)
        
        result = mock_dataset._pack_sequence(**sample_inputs)
        
        # Collect all indexes
        text_indexes = set(result['packed_text_indexes'].tolist())
        vit_indexes = set(result['packed_vit_token_indexes'].tolist())
        vae_indexes = set(result['packed_vae_token_indexes'].tolist())
        
        all_indexes = text_indexes | vit_indexes | vae_indexes
        
        # Should cover exactly 0 to sequence_length-1
        expected_indexes = set(range(result['sequence_length']))
        assert all_indexes == expected_indexes
        
        # No overlap between different token types
        assert len(text_indexes & vit_indexes) == 0
        assert len(text_indexes & vae_indexes) == 0 
        assert len(vit_indexes & vae_indexes) == 0