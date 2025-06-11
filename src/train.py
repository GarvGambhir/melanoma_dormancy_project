import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm # Use tqdm.notebook for Colab progress bars
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import wandb # Commented out for Weights & Biases

from src.config import Config
from src.data_loader import get_data_loaders
from src.model import MultiModalFusionTransformer

def train_model():
    # Initialize Weights & Biases - COMMENTED OUT
    # wandb.init(project=Config.WANDB_PROJECT, entity=Config.WANDB_ENTITY, config=vars(Config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data loaders
    train_loader, val_loader = get_data_loaders()

    # Determine gene input dimension from the first batch of the training data
    try:
        first_batch = next(iter(train_loader))
        gene_input_dim = first_batch['gene_expression'].shape[1]
        print(f"Detected gene input dimension: {gene_input_dim}")
    except StopIteration:
        print("Error: Training loader is empty. No data to determine gene input dimension.")
        return
    except Exception as e:
        print(f"Error determining gene input dimension: {e}")
        return


    model = MultiModalFusionTransformer(gene_input_dim=gene_input_dim, num_classes=Config.NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []

        # Gradient accumulation: Effectively increases batch size without requiring more GPU memory.
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            images = batch['image'].to(device)
            gene_expression = batch['gene_expression'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, gene_expression)
            loss = criterion(outputs, labels)
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS # Normalize loss for accumulation

            loss.backward()

            if (i + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS # Scale back for logging
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Perform a final optimizer step if the last batch didn't align with accumulation steps
        if (i + 1) % Config.GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        with torch.no_grad(): # Disable gradient calculations for validation
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                images = batch['image'].to(device)
                gene_expression = batch['gene_expression'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, gene_expression)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Validation F1: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")

        # Log to Weights & Biases - COMMENTED OUT
        # wandb.log({
        #     "train_loss": avg_train_loss,
        #     "train_accuracy": train_accuracy,
        #     "val_loss": avg_val_loss,
        #     "val_accuracy": val_accuracy,
        #     "val_f1": val_f1,
        #     "val_precision": val_precision,
        #     "val_recall": val_recall,
        #     "epoch": epoch
        # })

        # Save the model if it achieves the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(Config.MODELS_DIR, "best_melanoma_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with Val Accuracy: {best_val_accuracy:.4f} to {model_save_path}")

    print("Training finished.")
    # wandb.finish() # Commented out

