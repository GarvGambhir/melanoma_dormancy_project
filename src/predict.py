import torch
import torch.nn as nn
import os
import anndata as ad
from PIL import Image
import pandas as pd
import numpy as np
from src.config import Config
from src.model import MultiModalFusionTransformer
from src.utils.data_utils import image_transform

def predict_dormancy_reactivation(model_path, gene_input_dim, sample_data):
    """
    Makes a prediction for a single sample.
    Args:
        model_path (str): Path to the trained model's state_dict.
        gene_input_dim (int): The dimension of the gene expression input features.
        sample_data (dict): Dictionary containing 'gene_expression' (numpy array)
                            and 'image' (PIL Image) for a single sample.
    Returns:
        int: Predicted class label (0: Sensitive, 1: Dormant, 2: Reactivated).
        numpy.ndarray: Probabilities for each class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalFusionTransformer(gene_input_dim=gene_input_dim, num_classes=Config.NUM_CLASSES).to(device)
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode

    # Prepare inputs: Add a batch dimension (unsqueeze(0)) and move to device
    gene_expression_tensor = torch.tensor(sample_data['gene_expression'], dtype=torch.float32).unsqueeze(0).to(device)
    image_tensor = image_transform(sample_data['image']).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(image_tensor, gene_expression_tensor)
        probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
        predicted_class = probabilities.argmax(dim=1).item() # Get the class with highest probability

    return predicted_class, probabilities.cpu().squeeze(0).numpy()

# --- Example Usage for Prediction ---
# This part will be run directly in the notebook after the file is written.
# It assumes a model has been trained and processed data is available.
if __name__ == "__main__": # Keep this block for local testing if the file is run directly
    print("Running prediction script for testing...")

    # Define the path to the trained model
    model_path = os.path.join(Config.MODELS_DIR, "best_melanoma_model.pth")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first by running the training cell.")
    else:
        # Load the processed data manifest
        processed_manifest_path = os.path.join(Config.PROCESSED_DATA_DIR, "processed_manifest.csv")
        if not os.path.exists(processed_manifest_path):
            print("Processed data manifest not found. Please run the DataProcessor cell first.")
        else:
            manifest = pd.read_csv(processed_manifest_path)
            if len(manifest) == 0:
                print("Manifest is empty. No processed samples found to predict on.")
            else:
                # Pick the first processed sample for demonstration
                sample_row = manifest.iloc[0]
                print(f"Loading sample for prediction: {sample_row['sample_id']}")

                # Load actual processed data for this sample
                adata = ad.read_h5ad(sample_row['adata_path'])
                # Ensure gene expression is a dense numpy array for mean calculation
                gene_expression_data = adata.X.mean(axis=0).A.squeeze()
                image_data = Image.open(sample_row['image_path'])

                sample_data = {
                    'gene_expression': gene_expression_data,
                    'image': image_data
                }

                # Get the gene input dimension from the loaded data
                gene_input_dim = len(gene_expression_data)
                print(f"Using actual gene input dimension: {gene_input_dim}")

                # Make the prediction
                predicted_class, probabilities = predict_dormancy_reactivation(model_path, gene_input_dim, sample_data)

                # Map the predicted class back to meaningful names
                class_names = {v: k for k, v in Config.SAMPLE_STAGES.items()} # Reverse mapping
                predicted_class_name = class_names.get(predicted_class, "Unknown")

                print(f"\nPrediction for sample '{sample_row['sample_id']}':")
                print(f"  Predicted Class: {predicted_class_name}")
                print(f"  Probabilities: {probabilities}")
                print(f"  True Label (if available): {class_names.get(sample_row['label'], 'N/A')}")
