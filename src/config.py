import os

class Config:
    # Directories
    BASE_DIR = "/content/drive/MyDrive/melanoma_dormancy_project" 
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
    SRC_DIR = os.path.join(BASE_DIR, "src")

    # Data Specifics
    # Inferred from filenames: TC (Treatment-Sensitive), T0 (Dormant), T1-T4 (Reactivated stages)
    # Mapping these to numerical labels for classification
    # 0: Treatment-Sensitive, 1: Dormant, 2: Reactivated
    SAMPLE_STAGES = {
        "TC": 0,  # Treatment-Sensitive
        "T0": 1,  # Dormant
        "T1": 2,  # Reactivated
        "T2": 2,
        "T3": 2,
        "T4": 2
    }
    IMAGE_TYPES = ["tissue_hires_image.png.gz", "detected_tissue_image.jpg.gz"] # Prioritize hires
    GENE_EXPRESSION_FILES = ["matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"]
    POSITION_FILE = "tissue_positions_list.csv.gz"
    SCALE_FACTORS_FILE = "scalefactors_json.json.gz"

    # Model Parameters
    IMAGE_SIZE = (224, 224) # Standard for many pre-trained vision models
    GENE_EMBEDDING_DIM = 256 # Dimension for gene expression embeddings
    HIDDEN_DIM = 512 # Hidden dimension for transformer
    NUM_HEADS = 8
    NUM_LAYERS = 4 # Number of transformer encoder layers
    NUM_CLASSES = 3 # Sensitive, Dormant, Reactivated

    # Training Parameters
    BATCH_SIZE = 1 # Small batch size due to Colab free tier and data size
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10 # Increased slightly for better initial results, but keep low for prototype
    GRADIENT_ACCUMULATION_STEPS = 4 # To effectively increase batch size on limited GPU memory
    SAVE_MODEL_EVERY_EPOCH = False # Save only best for prototype
