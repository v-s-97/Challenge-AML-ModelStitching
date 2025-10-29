"""
Streamlined MLP Training and Submission Generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from src.common import TRAIN_DATA, TEST_DATA, DEVICE
from src.common import load_data, prepare_train_data, evaluate_retrieval, generate_submission

# Configuration
MODEL_PATH = "models/mlp_baseline.pth"
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.001
# 
class MLP(nn.Module):
    def __init__(self, input_dim=384, output_dim=768, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, device, epochs, lr):
    """Train the MLP model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
    
    return model

def main():
    print("="*60)
    print("MLP BASELINE TRAINING & SUBMISSION")
    print("="*60)
    
    # Load data
    print("\n1. Loading training data...")
    train_data = load_data(TRAIN_DATA)
    X, y = prepare_train_data(train_data)
    
    # Split train/val
    n_train = int(0.9 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\n2. Initializing model...")
    model = MLP().to(DEVICE)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n3. Training...")
    model = train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Evaluate on full training set
    print("\n4. Evaluating on training set...")
    caption_embd = train_data['caption_embd'].float()
    image_embd = train_data['img_embd'].float()
    gt_indices = train_data['caption2img_idx']
    
    for k in [1, 5, 10]:
        recall, mrr, l2_dist = evaluate_retrieval(
            model, caption_embd, image_embd, gt_indices, DEVICE, k=k
        )
        print(f"   Recall@{k}: {recall:.4f} | MRR: {mrr:.4f} | L2: {l2_dist:.4f}")
    
    # Generate submission
    print("\n5. Generating submission...")
    test_data = load_data(TEST_DATA)
    submission = generate_submission(model, test_data, DEVICE)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Submission saved to: submission.pt")


if __name__ == "__main__":
    main()
