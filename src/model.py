import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig # Using a pre-trained Vision Transformer
from src.config import Config # Import Config after it's written

class GeneEmbedding(nn.Module):
    """
    Simple embedding layer for gene expression data.
    Maps the high-dimensional gene expression vector to a lower-dimensional embedding.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.embedding(x)

class MultiModalFusionTransformer(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES,
                 gene_input_dim=None, # This will be set dynamically based on data
                 gene_embedding_dim=Config.GENE_EMBEDDING_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_heads=Config.NUM_HEADS,
                 num_layers=Config.NUM_LAYERS):
        super().__init__()

        if gene_input_dim is None:
            raise ValueError("gene_input_dim must be provided during model initialization.")

        # 1. Image Branch (Vision Transformer)
        # Using a pre-trained ViT for feature extraction.
        # 'google/vit-base-patch16-224-in21k' is a good starting point for medical images.
        # For faster prototyping, you can freeze the ViT weights. For fine-tuning, unfreeze.
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # self.freeze_vit_features() # Uncomment this line to freeze ViT weights for faster prototyping

        # Output dimension of ViT is typically 768 for the base model
        self.vit_output_dim = self.vit.config.hidden_size

        # 2. Gene Expression Branch
        self.gene_embedding = GeneEmbedding(gene_input_dim, gene_embedding_dim)

        # 3. Fusion Layer (Concatenation and Projection)
        # Combines ViT [CLS] token output and gene embedding into a single feature vector
        self.fusion_projection = nn.Sequential(
            nn.Linear(self.vit_output_dim + gene_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 4. Transformer Encoder for fusion
        # This layer learns complex interactions between the fused image and gene features.
        # We treat the single fused feature vector as a sequence of length 1 for the transformer.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Classification Head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def freeze_vit_features(self):
        """Freezes the parameters of the Vision Transformer to prevent fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("ViT feature extractor frozen.")

    def forward(self, images, gene_expression):
        # Process image data through the Vision Transformer
        # The ViT model expects 'pixel_values' as input
        vit_outputs = self.vit(pixel_values=images)
        # We take the output corresponding to the [CLS] token, which summarizes the image
        image_features = vit_outputs.last_hidden_state[:, 0, :]

        # Process gene expression data through the embedding layer
        gene_features = self.gene_embedding(gene_expression)

        # Concatenate image and gene features for fusion
        fused_features = torch.cat((image_features, gene_features), dim=1)
        # Project the concatenated features to the hidden dimension
        fused_features = self.fusion_projection(fused_features)

        # Add a sequence dimension for the transformer encoder (batch_size, 1, hidden_dim)
        # This allows us to use the TransformerEncoder even with a single fused token per sample.
        fused_features = fused_features.unsqueeze(1)
        transformer_output = self.transformer_encoder(fused_features)
        fused_features = transformer_output.squeeze(1) # Remove the sequence dimension

        # Pass the final fused features through the classification head
        logits = self.classifier(fused_features)
        return logits
