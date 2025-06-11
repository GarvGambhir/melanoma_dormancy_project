import os
import pandas as pd
import anndata as ad
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.config import Config
from src.utils.data_utils import image_transform # Import image_transform after it's written

class MelanomaDormancyDataset(Dataset):
    def __init__(self, manifest_df, processed_data_dir=Config.PROCESSED_DATA_DIR):
        self.manifest = manifest_df
        self.processed_data_dir = processed_data_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        sample_id = row['sample_id']
        adata_path = row['adata_path']
        image_path = row['image_path']
        label = row['label']

        # Load spatial transcriptomics data (AnnData)
        # For simplicity in prototyping, we'll average gene expression across all cells/spots in a sample.
        adata = ad.read_h5ad(adata_path)
        # Ensure gene expression is a dense numpy array for mean calculation
        gene_expression_features = torch.tensor(adata.X.mean(axis=0).A.squeeze(), dtype=torch.float32)

        # Load and transform histology image
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_transform(image)

        return {
            'sample_id': sample_id,
            'gene_expression': gene_expression_features,
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders(processed_data_dir=Config.PROCESSED_DATA_DIR, batch_size=Config.BATCH_SIZE):
    manifest_path = os.path.join(processed_data_dir, "processed_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Processed data manifest not found at {manifest_path}. Please run DataProcessor first.")

    full_manifest = pd.read_csv(manifest_path)

    # Split data into training and validation sets
    # Using stratify to ensure class distribution is maintained in splits
    train_manifest, val_manifest = train_test_split(
        full_manifest,
        test_size=0.2, # 20% for validation
        random_state=42, # for reproducibility
        stratify=full_manifest['label'] # Ensure balanced classes in splits
    )

    train_dataset = MelanomaDormancyDataset(train_manifest, processed_data_dir)
    val_dataset = MelanomaDormancyDataset(val_manifest, processed_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader
