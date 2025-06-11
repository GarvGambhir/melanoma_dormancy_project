import os
import pandas as pd
import numpy as np
import re # For regular expressions to parse filenames
from src.config import Config
from src.utils.data_utils import (
    load_spatial_transcriptomics_data,
    load_histology_image,
    load_scale_factors
)
import anndata as ad # Import anndata for saving

class DataProcessor:
    def __init__(self, raw_data_dir=Config.RAW_DATA_DIR, processed_data_dir=Config.PROCESSED_DATA_DIR):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def get_sample_groups(self):
        """
        Organizes raw data files by GSM ID and extracts sample stage from filenames.
        Assumes all .gz files are directly in the raw_data_dir.
        Returns a dictionary where keys are unique sample identifiers (e.g., 'WM4237_TC_GSM7845285')
        and values are lists of filenames belonging to that sample.
        """
        all_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.gz')]
        sample_groups = {}

        # Regex to extract GSM_ID and Sample_ID_Stage (e.g., WM4237_TC)
        # Pattern: GSMXXXXXXX_WMXXXX_STAGE_...
        pattern = r"^(GSM\d+)_([A-Z0-9]+_[A-Z0-9]+)_.*"

        for fname in all_files:
            match = re.match(pattern, fname)
            if match:
                gsm_id = match.group(1)
                sample_id_stage = match.group(2) # e.g., WM4237_TC, WM4007_T0
                unique_sample_key = f"{sample_id_stage}_{gsm_id}"

                if unique_sample_key not in sample_groups:
                    sample_groups[unique_sample_key] = []
                sample_groups[unique_sample_key].append(fname)
            else:
                print(f"Warning: Could not parse sample info from filename: {fname}. Skipping.")

        return sample_groups

    def process_all_data(self):
        """
        Processes all raw data samples and saves them in a structured format.
        """
        sample_groups = self.get_sample_groups()
        processed_samples_info = []

        print(f"Found {len(sample_groups)} unique samples to process.")

        for unique_sample_key, file_names_for_sample in sample_groups.items():
            print(f"Processing sample: {unique_sample_key}")

            # Extract the stage part (e.g., 'TC', 'T0', 'T1') from the unique_sample_key
            # The key is like 'WM4237_TC_GSM7845285', so we need the second part after the first underscore
            stage_str = unique_sample_key.split('_')[1]

            if stage_str not in Config.SAMPLE_STAGES:
                print(f"Warning: Unknown stage '{stage_str}' for sample {unique_sample_key}. Skipping.")
                continue

            label = Config.SAMPLE_STAGES[stage_str]

            # Load spatial transcriptomics data
            adata = load_spatial_transcriptomics_data(self.raw_data_dir, file_names_for_sample)
            if adata is None:
                print(f"Skipping {unique_sample_key} due to AnnData loading error.")
                continue

            # Load histology image (prioritize hires image)
            image = load_histology_image(self.raw_data_dir, file_names_for_sample, Config.IMAGE_TYPES[0])
            if image is None:
                print(f"Skipping {unique_sample_key} due to image loading error.")
                continue

            # Load scale factors (optional for prototype, but good to keep track)
            scale_factors = load_scale_factors(self.raw_data_dir, file_names_for_sample)
            # No need to skip if scale factors fail, as they are not directly used by the model yet

            # Save the processed data
            processed_adata_path = os.path.join(self.processed_data_dir, f"{unique_sample_key}_adata.h5ad")
            processed_image_path = os.path.join(self.processed_data_dir, f"{unique_sample_key}_image.png")

            try:
                adata.write(processed_adata_path)
                image.save(processed_image_path)
            except Exception as e:
                print(f"Error saving processed data for {unique_sample_key}: {e}. Skipping.")
                continue

            processed_samples_info.append({
                'sample_id': unique_sample_key,
                'adata_path': processed_adata_path,
                'image_path': processed_image_path,
                'label': label,
                'scale_factors': scale_factors # Store this if needed later
            })

        # Save a manifest of all processed files
        manifest_path = os.path.join(self.processed_data_dir, "processed_manifest.csv")
        pd.DataFrame(processed_samples_info).to_csv(manifest_path, index=False)
        print(f"\nProcessed data manifest saved to: {manifest_path}")
        print(f"Successfully processed {len(processed_samples_info)} samples.")

        return processed_samples_info

# Run the data processing
processor = DataProcessor()
processed_data_summary = processor.process_all_data()

# Display a sample of the processed data manifest
if processed_data_summary:
    print("\nFirst 5 entries in processed data manifest:")
    print(pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, "processed_manifest.csv")).head())
