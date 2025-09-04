import random
from typing import List, Tuple

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Handles vision feature extraction and projection. Supports ViT integration"""

    def __init__(
        self,
        vision_backbone_model: str,
        projector: nn.Module,
        freeze_backbone: bool = True,
        use_fast_processor: bool = True,
        sdxl_pooled_embed_dim: int = 1280,
        projector_hidden_state_index: int = -1,
        vision_feature_type: str = "openclip",
        openclip_model_name: str = "ViT-B-32",
        openclip_pretrained: str = "laion2b_s34b_b79k",
    ):
        super().__init__()
        self.vision_feature_type = vision_feature_type
        if vision_feature_type == "openclip":
            self.encoder = OpenCLIPVisionEncoder(
                model_name=openclip_model_name,
                pretrained=openclip_pretrained,
                projector=projector,
                freeze_backbone=freeze_backbone,
                sdxl_pooled_embed_dim=sdxl_pooled_embed_dim,
            )
        else:
            self.processor = AutoImageProcessor.from_pretrained(vision_backbone_model, use_fast=use_fast_processor)
            self.vit = AutoModel.from_pretrained(vision_backbone_model)
            self.projector = projector
            self.sdxl_pooled_embed_dim = sdxl_pooled_embed_dim
            self.projector_hidden_state_index = projector_hidden_state_index
            self.pooler_projector = self._create_pooler_projector()
            if freeze_backbone:
                self._freeze_backbone()

    def _validate_inputs(self, vision_backbone_model: str, projector: nn.Module) -> None:
        """Validate constructor inputs."""
        if not vision_backbone_model or not vision_backbone_model.strip():
            raise ConfigurationError("Vision backbone model name cannot be empty")
        if projector is None:
            raise ConfigurationError("Projector cannot be None")

    def _create_pooler_projector(self) -> nn.Linear:
        """Create and initialize the pooler projector."""
        # Ensure vit_hidden_dim is always an int (not a tuple)
        vit_hidden_dim = self.vit.config.hidden_size
        if isinstance(vit_hidden_dim, (tuple, list)):
            # Some models may return (H, W), take the product or first element as appropriate
            if len(vit_hidden_dim) == 1:
                vit_hidden_dim = vit_hidden_dim[0]
            else:
                # If it's a tuple of two, assume patch grid and use the last hidden size
                vit_hidden_dim = vit_hidden_dim[-1]
        pooler_projector = nn.Linear(vit_hidden_dim, self.sdxl_pooled_embed_dim)
        ProjectorInitializer.initialize_with_zeros(pooler_projector)
        return pooler_projector

    def _freeze_backbone(self) -> None:
        """Freeze the vision transformer backbone."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def encode(self, images: Union[List[Image.Image], torch.Tensor]) -> EmbeddingBundle:
        if hasattr(self, "encoder"):
            return self.encoder.encode(images)

        # Extract and project vision features from input images.
        vit_outputs = self._extract_vit_features(images)
        sequence_embeds = self._project_sequence_features(vit_outputs.hidden_states)
        pooled_embeds = self._project_pooled_features(vit_outputs.pooler_output)

        return EmbeddingBundle(sequence_embeds, pooled_embeds)

    def _extract_vit_features(self, images: Union[List[Image.Image], torch.Tensor]):
        """Extract features from images using ViT."""
        vit_inputs = self._prepare_vit_inputs(images)
        return self.vit(**vit_inputs, output_hidden_states=True)

    def _prepare_vit_inputs(self, images: Union[List[Image.Image], torch.Tensor]):
        """Prepare inputs for ViT processing."""
        should_rescale = not isinstance(images, torch.Tensor)
        vit_inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_rescale=should_rescale,
        )
        return vit_inputs.to(self.vit.device)

