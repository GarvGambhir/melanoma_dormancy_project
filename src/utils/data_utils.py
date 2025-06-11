import os
import gzip
import pandas as pd
import numpy as np
from PIL import Image
import json
import io
from scipy.io import mmread
import anndata as ad
import torch
import torchvision.transforms as T
from src.config import Config # Import Config after it's written

# Image transformation for the model
image_transform = T.Compose([
    T.Resize(Config.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet standards
])

def decompress_and_read_gz(filepath, mode='r'):
    """Decompresses a .gz file and reads its content."""
    with gzip.open(filepath, mode) as f:
        if mode == 'rb': # For images
            return f.read()
        else: # For text files
            return f.read().decode('utf-8')

def find_file_in_list(file_list, keyword):
    """Helper to find a file containing a keyword in a list of filenames."""
    for f in file_list:
        if keyword in f:
            return f
    return None

def load_spatial_transcriptomics_data(base_dir, file_names):
    """
    Loads spatial transcriptomics data for a single sample given a list of its filenames.
    Returns an anndata object.
    """
    try:
        matrix_fname = find_file_in_list(file_names, 'matrix.mtx.gz')
        barcodes_fname = find_file_in_list(file_names, 'barcodes.tsv.gz')
        features_fname = find_file_in_list(file_names, 'features.tsv.gz')
        positions_fname = find_file_in_list(file_names, 'tissue_positions_list.csv.gz')

        if not all([matrix_fname, barcodes_fname, features_fname, positions_fname]):
            raise FileNotFoundError("One or more required spatial transcriptomics files not found.")

        matrix_path = os.path.join(base_dir, matrix_fname)
        barcodes_path = os.path.join(base_dir, barcodes_fname)
        features_path = os.path.join(base_dir, features_path)
        positions_path = os.path.join(base_dir, positions_path)

        # Decompress and read
        matrix_data = mmread(io.BytesIO(decompress_and_read_gz(matrix_path, 'rb')))
        barcodes_df = pd.read_csv(io.StringIO(decompress_and_read_gz(barcodes_path)), header=None, sep='\t')
        features_df = pd.read_csv(io.StringIO(decompress_and_read_gz(features_path)), header=None, sep='\t')
        positions_df = pd.read_csv(io.StringIO(decompress_and_read_gz(positions_path)), header=None)

        # Create AnnData object
        # The matrix from 10X Genomics is usually genes x cells, transpose to cells x genes
        adata = ad.AnnData(X=matrix_data.transpose().tocsr())
        adata.obs_names = barcodes_df[0].values # Cell barcodes
        adata.var_names = features_df[1].values # Gene names (use column 1 for human readable)

        # Make var_names unique
        adata.var_names_make_unique() # Add this line

        # Add spatial coordinates to .obs
        # Assuming format from 10X Genomics: barcode, in_tissue, array_row, array_col, pixel_row, pixel_col
        positions_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pixel_row', 'pixel_col']
        positions_df = positions_df.set_index('barcode')
        adata.obs = adata.obs.merge(positions_df, left_index=True, right_index=True, how='left')

        # Add spatial coordinates to .obsm['spatial']
        adata.obsm['spatial'] = adata.obs[['pixel_row', 'pixel_col']].values

        return adata
    except Exception as e:
        print(f"Error loading spatial transcriptomics data: {e}")
        return None

def load_histology_image(base_dir, file_names, image_type="tissue_hires_image.png.gz"):
    """
    Loads a histology image for a single sample given a list of its filenames.
    """
    try:
        image_fname = find_file_in_list(file_names, image_type)
        if not image_fname:
            raise FileNotFoundError(f"Image file '{image_type}' not found in the list.")
        image_path = os.path.join(base_dir, image_fname)
        img_bytes = decompress_and_read_gz(image_path, 'rb')
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_scale_factors(base_dir, file_names):
    """
    Loads scale factors JSON for a single sample given a list of its filenames.
    """
    try:
        scale_factors_fname = find_file_in_list(file_names, Config.SCALE_FACTORS_FILE)
        if not scale_factors_fname:
            raise FileNotFoundError(f"Scale factors file '{Config.SCALE_FACTORS_FILE}' not found.")
        scale_factors_path = os.path.join(base_dir, scale_factors_fname)
        json_bytes = decompress_and_read_gz(scale_factors_path, 'rb')
        return json.loads(json_bytes.decode('utf-8'))
    except Exception as e:
        print(f"Error loading scale factors: {e}")
        return None
