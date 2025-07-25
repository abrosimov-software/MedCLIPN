import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from typing import Optional


class CLIPNEmbeddingAdapter(nn.Module):
    """
    Self-contained embedding adapter that handles all CLIPN "no" prompt logic.
    This module encapsulates the original embedding layer and adds CLIPN functionality.
    """
    
    def __init__(self, original_embedding, num_no_texts=16, context_length=77, std=0.01):
        super().__init__()
        self.embedding = original_embedding
        self.num_no_texts = num_no_texts
        self.context_length = context_length
        
        # Initialize learnable "no" prompt embeddings
        self.prompt_no = nn.Parameter(
            torch.zeros(num_no_texts, context_length, original_embedding.embedding_dim)
        )
        
        # Initialize parameters
        nn.init.normal_(self.prompt_no, std=std)
    
    def forward(self, input):
        """Forward pass with integrated prompt processing."""
        # Get original embeddings
        embeddings = self.embedding(input)
        
        # Apply "no" prompt processing
        return self._process_text_with_no_prompt(embeddings)
    
    def _process_text_with_no_prompt(self, text_embeddings):
        """Process text embeddings with "no" prompt during training and inference."""
        batch_size = text_embeddings.size(0)
        
        if self.training:
            # Random selection during training
            idx = np.random.randint(0, self.num_no_texts + 1, (batch_size,))
            prompt_no = torch.cat([
                self.prompt_no, 
                torch.mean(self.prompt_no, dim=0, keepdim=True)
            ], 0)[idx]
            return text_embeddings + prompt_no
        else:
            # Use mean of all prompts during inference
            prompt_no = torch.mean(self.prompt_no, dim=0, keepdim=True)
            return text_embeddings + prompt_no
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the original embedding."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ["embedding", "prompt_no"]:
                raise
            return getattr(self.embedding, name)
        

class CLIPNAdapter(nn.Module):
    """
    CLIPN adapter that adds "no" text encoding capability to existing CLIP models.
    Preserves original model functionality while adding CLIPN-style negative text encoding.
    """

    def __init__(self, base_clip_model, tokenizer, num_no_texts=16, frozen=True, **kwargs):
        super().__init__()
        self.base_model = base_clip_model
        self.num_no_texts = num_no_texts
        self.tokenizer = tokenizer
        self.frozen = frozen
        
        # Initialize "no" text encoder by copying entire base model
        self._init_no_submodule()
        
        # Replace embedding layer with self-contained CLIPN embedding adapter
        self._replace_embedding_layer()
        
        # Freeze original model weights
        self._freeze_original_model()
        
        # Unfreeze visual encoder if specified
        if not self.frozen:
            self._unfreeze_visual_encoder()
    
    def _init_no_submodule(self):
        """Initialize the "no" text encoder by copying entire base model."""
        self.model_no = deepcopy(self.base_model)
        
        # Remove visual component to save memory
        if hasattr(self.model_no, 'visual'):
            delattr(self.model_no, 'visual')
    
    def _replace_embedding_layer(self):
        """Replace the embedding layer in the "no" model with CLIPN embedding adapter."""
        original_embedding = self._find_embedding_layer(self.model_no)
        if original_embedding is None:
            raise RuntimeError("No embedding layer found in the model")
        
        # Create self-contained CLIPN embedding adapter
        clipn_embedding = CLIPNEmbeddingAdapter(
            original_embedding=original_embedding,
            num_no_texts=self.num_no_texts,
            context_length=self.tokenizer.context_length,
            std=0.01
        )
        
        # Replace the embedding layer
        self._set_embedding_layer(self.model_no, clipn_embedding)
    
    def _find_embedding_layer(self, module):
        """Find and return the first embedding layer in the module."""
        for child in module.modules():
            if isinstance(child, nn.Embedding):
                return child
        return None
    
    def _set_embedding_layer(self, module, new_embedding):
        """Recursively find and replace the first embedding layer."""
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding):
                setattr(module, name, new_embedding)
                return True
            else:
                if self._set_embedding_layer(child, new_embedding):
                    return True
        return False
    
    def _freeze_original_model(self):
        """Freeze all parameters of the original model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _unfreeze_visual_encoder(self):
        """Unfreeze all parameters of the visual encoder."""
        if hasattr(self.base_model, 'visual'):
            for param in self.base_model.visual.parameters():
                param.requires_grad = True
    
    def encode_text_no(self, text, normalize=True):
        """Encode text using the "no" encoder with learnable prompt embeddings."""
        return self.model_no.encode_text(text, normalize)
    
    def forward(self, image, text, **kwargs):
        """Forward pass returning original features plus "no" text features."""
        if text is None:
            return self.base_model.encode_image(image)
            
        # Call original model methods
        image_features = self.base_model.encode_image(image, normalize=True)
        text_features = self.base_model.encode_text(text, normalize=True)
        
        # Call new "no" text encoder
        text_features_no = self.encode_text_no(text, normalize=True)
        
        # Return all features and logit scale
        return image_features, text_features, text_features_no, self.base_model.logit_scale.exp()
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ["base_model", "model_no"]:
                raise
            return getattr(self.base_model, name)



class CLIPNTransformerAdapter(nn.Module):
    """
    Self-contained transformer adapter that handles all CLIPN "no" prompt logic.
    This module encapsulates the original transformer and adds CLIPN functionality.
    """
    
    def __init__(self, original_transformer, num_no_texts=16, context_length=77, std=0.01):
        super().__init__()
        self.transformer = original_transformer
        self.num_no_texts = num_no_texts
        self.context_length = context_length
        
        # Infer embedding dimension from the original transformer
        self.embedding_dim = self._infer_embedding_dim(original_transformer)
        
        # Initialize learnable "no" prompt embeddings
        self.prompt_no = nn.Parameter(
            torch.zeros(num_no_texts, context_length, self.embedding_dim)
        )
    
        # Initialize parameters
        nn.init.normal_(self.prompt_no, std=std)
    
    def _infer_embedding_dim(self, transformer):
        """Infer embedding dimension from the transformer layer."""
        
        # Try to get from width attribute (common in CLIP)
        if hasattr(transformer, 'width'):
            return transformer.width
        
        # Try to get from config (common in HuggingFace models)
        if hasattr(transformer, 'config'):
            config = transformer.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
            if hasattr(config, 'n_embd'):
                return config.n_embd
            if hasattr(config, 'd_model'):
                return config.d_model
            
        # Try to get from transformer's first layer
        if hasattr(transformer, 'layers') and len(transformer.layers) > 0:
            first_layer = transformer.layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'embed_dim'):
                return first_layer.self_attn.embed_dim
        
        # Try to get from resblocks (common in CLIP)
        if hasattr(transformer, 'resblocks') and len(transformer.resblocks) > 0:
            first_block = transformer.resblocks[0]
            if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'embed_dim'):
                return first_block.attn.embed_dim
        
        # Default fallback
        return 512
    
    def forward(self, x, *args, **kwargs):
        """Forward pass with integrated prompt processing."""
        # Apply "no" prompt processing first
        x = self._process_text_with_no_prompt(x)
            
        # Pass through original transformer
        return self.transformer(x, *args, **kwargs)
    
    def _process_text_with_no_prompt(self, text_embeddings):
        """Process text embeddings with "no" prompt during training and inference."""
        batch_size = text_embeddings.size(0)
        
        if self.training:
            # Random selection during training
            idx = np.random.randint(0, self.num_no_texts + 1, (batch_size,))
            prompt_no = torch.cat([
                self.prompt_no, 
                torch.mean(self.prompt_no, dim=0, keepdim=True)
            ], 0)[idx]
            return text_embeddings + prompt_no
        else:
            # Use mean of all prompts during inference
            prompt_no = torch.mean(self.prompt_no, dim=0, keepdim=True)
            return text_embeddings + prompt_no
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the original transformer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ["transformer", "prompt_no"]:
                raise
            return getattr(self.transformer, name)
        

class AltCLIPNAdapter(nn.Module):
    """
    CLIPN adapter that adds "no" text encoding capability to existing CLIP models.
    Preserves original model functionality while adding CLIPN-style negative text encoding.
    """

    def __init__(self, base_clip_model, tokenizer, transformer_path, num_no_texts=16, frozen=True, **kwargs):
        super().__init__()
        self.base_model = base_clip_model
        self.num_no_texts = num_no_texts
        self.tokenizer = tokenizer
        self.frozen = frozen
        self.transformer_path = transformer_path
        
        # Initialize "no" text encoder by copying entire base model
        self._init_no_submodule()
        
        # Replace transformer layer with self-contained CLIPN transformer adapter
        if self.num_no_texts > 0:
            self._replace_transformer_layer()
        
        # Freeze original model weights
        self._freeze_original_model()
        
        # Unfreeze visual encoder if specified
        if not self.frozen:
            self._unfreeze_visual_encoder()
    
    def _init_no_submodule(self):
        """Initialize the "no" text encoder by copying entire base model."""
        self.model_no = deepcopy(self.base_model)
        
        # Remove visual component to save memory
        if hasattr(self.model_no, 'visual'):
            delattr(self.model_no, 'visual')
    
    def _replace_transformer_layer(self):
        """Replace the transformer layer in the "no" model with CLIPN transformer adapter."""
        original_transformer, parent_module, attr_name = self._get_transformer_by_path(self.model_no, self.transformer_path)
        if original_transformer is None:
            raise RuntimeError(f"No transformer layer found at path: {self.transformer_path}")
        
        # Create CLIPN transformer adapter
        clipn_transformer = CLIPNTransformerAdapter(
            original_transformer=original_transformer,
            num_no_texts=self.num_no_texts,
            context_length=self.tokenizer.context_length,
            std=0.01
        )
        
        # Replace the transformer layer
        setattr(parent_module, attr_name, clipn_transformer)
    
    def _get_transformer_by_path(self, module, path):
        """Get transformer layer by user-specified path."""
        path_parts = path.split('.')
        
        # Navigate to the parent module
        current_module = module
        for part in path_parts[:-1]:
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                return None, None, None
        
        # Get the final attribute name and the transformer
        final_attr = path_parts[-1]
        if hasattr(current_module, final_attr):
            transformer = getattr(current_module, final_attr)
            return transformer, current_module, final_attr
        else:
            return None, None, None
    
    def _freeze_original_model(self):
        """Freeze all parameters of the original model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _unfreeze_visual_encoder(self):
        """Unfreeze all parameters of the visual encoder."""
        if hasattr(self.base_model, 'visual'):
            for param in self.base_model.visual.parameters():
                param.requires_grad = True
    
    def encode_text_no(self, text, normalize=True):
        """Encode text using the "no" encoder with learnable prompt embeddings."""
        return self.model_no.encode_text(text, normalize)
    
    def forward(self, image, text, **kwargs):
        """Forward pass returning original features plus "no" text features."""
        if text is None:
            return self.base_model.encode_image(image)
            
        # Call original model methods
        image_features = self.base_model.encode_image(image, normalize=True)
        text_features = self.base_model.encode_text(text, normalize=True)
        
        # Call new "no" text encoder
        text_features_no = self.encode_text_no(text, normalize=True)
        
        # Return all features and logit scale
        return image_features, text_features, text_features_no, self.base_model.logit_scale.exp()
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ["base_model", "model_no"]:
                raise
            return getattr(self.base_model, name)