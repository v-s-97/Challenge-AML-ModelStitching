import argparse
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

# roberta-large-nli-stsb-mean-tokens
def load_text_model(model_name="sentence-transformers/roberta-large-nli-stsb-mean-tokens"):
    """Load Sentence-BERT text encoder."""
    print(f"Loading text model: {model_name}")
    return SentenceTransformer(model_name)


def load_image_model(model_name="facebook/dinov2-giant"):
    """Load DINOv2 image encoder."""
    print(f"Loading image model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    return image_processor, model


@torch.inference_mode()
def process_images_batch(image_processor, model, image_paths, device, batch_size=128, dataset_path=None):
    """Generate image embeddings in batches."""
    print(f"Processing {len(image_paths)} images in batches...")
    model.to(device)
    model.eval()
    
    all_embeddings = []
    failed_indices = []
    img_files = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        valid_images = []
        
        # Keep track of which original indices correspond to valid images in the batch
        valid_paths = []

        for j, path in enumerate(batch_paths):
            original_index = i + j
            try:
                img = Image.open(dataset_path / 'Images' / path).convert("RGB")
                valid_images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Skipping image {path} due to error: {e}")
                failed_indices.append(original_index)

        if not valid_images:
            continue

        inputs = image_processor(images=valid_images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        # Average over patch tokens
        image_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Store embeddings based on their success
        all_embeddings.extend(image_features)
        img_files.extend(valid_paths)
    
    if not all_embeddings:
        return np.array([]), list(range(len(image_paths)))

    return img_files, np.vstack(all_embeddings)

def process_captions(text_model, captions, device):
    """Generate text embeddings using Sentence-BERT."""
    print("Processing captions...")
    return text_model.encode(
        captions, 
        convert_to_numpy=True, 
        show_progress_bar=True, 
        device=device
    )

def load_dataset(dataset_path):
    """
    Load dataset from a directory containing captions.txt and an Images folder.
    """
    captions_file = dataset_path / "captions.txt"
    images_dir = dataset_path / "Images"

    if not captions_file.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Could not find 'captions.txt' or 'Images' directory in {dataset_path}")

    df = pd.read_csv(captions_file)
    
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df))

    return df

def create_data_file(dataset_path, output_file, device=None, args={}):
    """
    Main function to generate embeddings and save the final .npz file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    text_model = load_text_model()
    image_processor, image_model = load_image_model()

    print(f"Loading dataset from: {dataset_path}")
    df_captions = load_dataset(dataset_path)
    all_captions = df_captions['caption'].tolist()
    caption2img = df_captions['image'].tolist()
    all_images = df_captions['image'].unique().tolist()

    num_images = len(all_images)
    num_captions = len(all_captions)
    print(f"Found {num_images} images and {num_captions} total captions.")

    all_images, img_embd = process_images_batch(image_processor, image_model, all_images, device, dataset_path=dataset_path)
    images_dict = {img_name: i for i, img_name in enumerate(all_images)}

    caption_embeddings = process_captions(text_model, df_captions['caption'].tolist(), device)
    
    label = np.zeros((num_captions, num_images), dtype=np.bool)
    for idx in range(num_captions):
        img_name = caption2img[idx]
        img_idx = images_dict[img_name]
        label[idx, img_idx] = 1

    data = {
        'metadata/num_captions': np.array([num_captions]),
        'metadata/num_images': np.array([num_images]),
        'metadata/embedding_dim_text': np.array([caption_embeddings.shape[1]]),
        'metadata/embedding_dim_image': np.array([img_embd.shape[1]]),
        'captions/ids': df_captions['id'].to_numpy(),
        'captions/text': np.array(all_captions),
        'captions/embeddings': caption_embeddings,
        'captions/label': label,
        'images/names': np.array(all_images),
        'images/embeddings': img_embd,
    }

    print(f"Saving processed data to {output_file}")
    np.savez_compressed(output_file, **data)
    print("✓ Done.")
    
    if args.create_secret_version:
        data_secret = {
            'metadata/num_captions': np.array([num_captions]),
            'metadata/embedding_dim_text': np.array([caption_embeddings.shape[1]]),
            'metadata/embedding_dim_image': np.array([img_embd.shape[1]]),
            'captions/ids': df_captions['id'].to_numpy(),
            'captions/text': np.array(all_captions),
            'captions/embeddings': caption_embeddings,
        }

        secret_output_file = str(Path(output_file).with_suffix('.clean.npz'))
        print(f"Saving secret version to {secret_output_file}")
        np.savez_compressed(secret_output_file, **data_secret)
        print("✓ Secret version saved.")
        

def main():
    parser = argparse.ArgumentParser(description="Preprocess image-caption dataset and save to a .npz file.", add_help=True)
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the dataset folder (e.g., 'data/train')."
    )
    parser.add_argument(
        "--output-file", '-o',
        type=str,
        default="processed_data.npz",
        help="Path to save the output .npz file."
    )
    parser.add_argument(
        "--create-secret-version",
        action='store_true',
        help="Create a secret version of the output file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Autodetects if not specified."
    )
    args = parser.parse_args()

    create_data_file(args.input_folder, args.output_file, args.device, args)

if __name__ == "__main__":
    main()
