import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_data(path):
    """Load processed data from .npz file"""
    data = dict(np.load(path, allow_pickle=True))
    # data['caption2img'] = data['caption2img'].item()
    # data['caption2img_idx'] = data['caption2img_idx'].item()
    return data

def prepare_train_data(data):
    """Prepare training data from loaded dict"""
    caption_embd = data['captions/embeddings']
    image_embd = data['images/embeddings']
    # Map caption embeddings to corresponding image embeddings
    label = data['captions/label'] # N x M

    # repeat the image embeddings according to the label
    label_idx = np.nonzero(label)[1]
    print(label_idx.shape)
    image_embd = image_embd[label_idx]
    assert caption_embd.shape[0] == image_embd.shape[0], "Mismatch in number of caption and image embeddings"

    X = torch.from_numpy(caption_embd).float()
    # Map each caption to its corresponding image embedding
    y = torch.from_numpy(image_embd).float()
    label = torch.from_numpy(label).bool()

    print(f"Train data: {len(X)} captions, {len(image_embd)} images")
    return X, y, label

def generate_submission(sample_ids, translated_embeddings, output_file="submission.csv"):
    """
    Generate a submission.csv file from translated embeddings.
    """
    print("Generating submission file...")

    if isinstance(translated_embeddings, torch.Tensor):
        translated_embeddings = translated_embeddings.cpu().numpy()

    # Create a DataFrame with sample_id and embeddings

    df_submission = pd.DataFrame({'id': sample_ids, 'embedding': translated_embeddings.tolist()})

    df_submission.to_csv(output_file, index=False, float_format='%.17g')
    print(f"✓ Saved submission to {output_file}")
    
    return df_submission