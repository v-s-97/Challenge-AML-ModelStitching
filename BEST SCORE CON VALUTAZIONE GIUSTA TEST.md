```python
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



from challenge.src.common import load_data, prepare_train_data, generate_submission
from challenge.src.eval import evaluate_retrieval, visualize_retrieval
# ==== Config ====
MODEL_PATH = "models/maxmatch_adapter_k6_sinkhorn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 120
BATCH_SIZE = 256



LR = 0.0012
WEIGHT_DECAY = 5e-5

K_SLOTS = 6
SLOT_DROPOUT_P = 0.15

SINKHORN_ITERS = 11
SINKHORN_TAU_START = 0.208
SINKHORN_TAU_END   = 0.190
DETACH_ASSIGNMENT = True

SCALE_S = 0.47
DELTA_1 = 0.38
DELTA_2 = 0.61
DELTA_3 = 0.564

LAMBDA_ISDL = 0.110
LAMBDA_GDL  = 0.085
LAMBDA_MMD  = 0.015
LAMBDA_DIV  = 0.019





# ============================================================
# 1) Head set-based: t(1024) -> S_T (K x 1536)
# ============================================================
class SetPredictionHead(nn.Module):
    """
    Converte un embedding testuale globale in K slot nello spazio visivo (1536).
    - K query vettori learnable (inizializzati nel target space).
    - Proiezione testo -> spazio visivo + gating per diversificare contributi.
    - Piccolo dropout per evitare collasso precoce.
    """
    def __init__(self, d_text=1024, d_vis=1536, K=4, hidden=2048, slot_dropout_p=0.1):
        super().__init__()
        self.K = K
        self.d_vis = d_vis

        self.text_to_vis = nn.Sequential(
            nn.Linear(d_text, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_vis),
        )

        self.slot_queries = nn.Parameter(torch.randn(K, d_vis) * 0.02)
        self.gate_per_slot = nn.Linear(d_text, K)
        self.delta_per_slot = nn.Linear(d_text, K * d_vis)

        self.ln_slots = nn.LayerNorm(d_vis)
        self.ln_text  = nn.LayerNorm(d_vis)
        self.dropout = nn.Dropout(p=slot_dropout_p)

    def forward(self, t: torch.Tensor):
        B = t.size(0)
        t_vis = self.text_to_vis(t)                 # (B, d_vis)
        t_vis = self.ln_text(t_vis)

        gate  = torch.sigmoid(self.gate_per_slot(t))      # (B, K)
        delta = self.delta_per_slot(t).view(B, self.K, self.d_vis)

        Q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_vis)
        t_vis_exp = t_vis.unsqueeze(1).expand(-1, self.K, -1)
        gate_exp  = gate.unsqueeze(-1)

        # --- residui (pre-fusione globale) ---
        R = Q + gate_exp * t_vis_exp + delta               # (B, K, d_vis)
        R = self.ln_slots(R)
        R = self.dropout(R)
        E_T = F.normalize(R, dim=-1)                       # residui normalizzati (per ISDL)

        # --- fusione globale per scoring ---
        S_T = E_T + F.normalize(t_vis, dim=-1).unsqueeze(1)
        S_T = F.normalize(S_T, dim=-1)
        return S_T, E_T, F.normalize(t_vis, dim=-1)
# ============================================================
# 2) Sinkhorn matching (soft, entropic-regularized doubly-stochastic)
#    sim -> P ~ doubly-stochastic, poi S_H = <P, sim>/K
# ============================================================
def sinkhorn_logspace(log_K: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """
    Log-space Sinkhorn per stabilità numerica.
    log_K: (B, K, K) log-kernel (logits pre-softmax)
    Ritorna log_P: (B, K, K) ~ log matrix bistocastica
    """
    B, K, _ = log_K.shape
    log_u = torch.zeros(B, K, device=log_K.device)
    log_v = torch.zeros(B, K, device=log_K.device)

    for _ in range(iters):
        # normalizza righe
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        # normalizza colonne
        log_v = -torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

    log_P = log_K + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    return log_P

def sinkhorn_assignment(sim: torch.Tensor, tau: float = 0.1, iters: int = 10,
                        detach_input: bool = True) -> torch.Tensor:
    """
    sim: (B, K, K) similarità coseno.
    Costruisce kernel K_ij = exp(sim_ij / tau), applica Sinkhorn per ottenere P ~ bistocastica.
    Se detach_input=True, rimuove il gradiente dalla matrice di sim nel calcolo dell'assegnamento,
    replicando lo schema "stop-grad" usato in MaxMatch per la parte di matching.
    """
    if detach_input:
        sim = sim.detach()
    log_K = sim / max(tau, 1e-6)
    log_P = sinkhorn_logspace(log_K, iters=iters)
    P = torch.exp(log_P)  # (B, K, K), righe/colonne ~ 1
    return P

def cosine_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # (B, K, D) x (B, D, K) -> (B, K, K)
    return torch.matmul(A, B.transpose(-1, -2))

def s_h_maxmatch_sinkhorn(
    S_T: torch.Tensor, V: torch.Tensor, *,
    tau: float, iters: int, detach: bool
) -> torch.Tensor:
    B, K, D = S_T.shape
    Vn = F.normalize(V, dim=-1)
    S_V = Vn.unsqueeze(1).expand(-1, K, -1)        # (B, K, D)
    sims = torch.matmul(S_T, S_V.transpose(-1, -2))# (B, K, K)
    P = sinkhorn_assignment(sims, tau=tau, iters=iters, detach_input=detach)
    return (P * sims).sum(dim=(1, 2)) / K
    
USE_DEGENERATE_SH = True  # True: rapido e identico nel caso colonne uguali

def s_h_singleton_target(S_T: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # S_T: (B,K,D), V: (B,D)
    Vn = F.normalize(V, dim=-1)
    sims = torch.einsum('bkd,bd->bk', S_T, Vn)  # (B,K)
    return sims.mean(dim=1)                      # (B,)

def s_h(S_T: torch.Tensor, V: torch.Tensor, *, tau, iters, detach) -> torch.Tensor:
    if USE_DEGENERATE_SH:
        return s_h_singleton_target(S_T, V)
    else:
        return s_h_maxmatch_sinkhorn(S_T, V, tau=tau, iters=iters, detach=detach)



def triplet_maxmatch_sh(S_T, V, delta1, *, tau, iters, detach):
    B, K, D = S_T.shape
    s_pos = s_h_maxmatch_sinkhorn(S_T, V, tau=tau, iters=iters, detach=detach)  # (B,)
    V_all = F.normalize(V, dim=-1)

    max_negs = []
    CH = 128  # blocco sicuro; puoi alzare/abbassare in base alla GPU
    for start in range(0, B, CH):
        end = min(B, start + CH)
        S_blk = S_T[start:end]                                   # (ch,K,D)

        # confronta ogni S_blk[i] con tutte le immagini del batch (B)
        S_exp = S_blk.unsqueeze(1).expand(end - start, B, K, D).reshape((end - start) * B, K, D)
        V_exp = V_all.unsqueeze(0).expand(end - start, B, D).reshape((end - start) * B, D)

        s_blk = s_h(S_exp, V_exp, tau=tau, iters=iters, detach=detach)

        s_blk = s_blk.view(end - start, B)                       # (ch, B)

        # maschera SOLO il positivo per riga: colonna (start + r)
        rows = torch.arange(end - start, device=S_T.device)
        cols = torch.arange(start, end, device=S_T.device)
        s_blk[rows, cols] = float('-inf')

        max_negs.append(s_blk.max(dim=1).values)                 # (ch,)

    s_neg = torch.cat(max_negs, dim=0)                           # (B,)
    return F.relu(delta1 + s_neg - s_pos).mean()

# ============================================================
# 3) ISDL – Intra-Set Diversity Loss (Alomari 2025)
#     Minimizza similarità intra-slot (promuove diversità)
# ============================================================
def isdl_intra_set_diversity_exp(S_T: torch.Tensor, s: float, delta3: float) -> torch.Tensor:
    B, K, D = S_T.shape
    C = torch.matmul(S_T, S_T.transpose(-1, -2)).clamp(-1, 1)  # (B,K,K)
    mask = ~torch.eye(K, device=S_T.device, dtype=torch.bool).unsqueeze(0).expand(B, K, K)
    C_off = C[mask]  # (B*K*(K-1),)
    loss = torch.exp(s * (C_off - delta3)).mean()
    return loss

import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===========================
# 0) Seed e backend (come sopra)
# ===========================
import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===========================
# 1) Caricamento dati (caption/image embeddings)
# ===========================
train_data = load_data("data/train/train.npz")
X, y, label = prepare_train_data(train_data)   # X: (N_cap, D_txt)  | y: (N_cap, D_img)

# ===========================
# 2) Split a LIVELLO IMMAGINE (prima del training)
# ===========================
import hashlib
from scipy import sparse

img_names_all = train_data['images/names']                                 # (N_img,)
img_emb_all   = torch.from_numpy(train_data['images/embeddings']).float()  # (N_img, D_img)

val_ratio = 0.10
def stable_hash(name: str) -> float:
    h = hashlib.md5(name.encode('utf-8')).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

img_hash       = np.array([stable_hash(nm) for nm in img_names_all])
IMG_VAL_MASK   = (img_hash < val_ratio)           # np.bool_
IMG_TRAIN_MASK = ~IMG_VAL_MASK

# Mappa caption → indice globale dell'immagine GT
cap_to_img = train_data['captions/label']
if sparse.issparse(cap_to_img):
    cap_gt_img_idx = cap_to_img.argmax(axis=1).A1
else:
    cap_gt_img_idx = np.argmax(cap_to_img, axis=1)

# Maschere caption coerenti con lo split immagine
CAP_TRAIN_MASK = IMG_TRAIN_MASK[cap_gt_img_idx]   # np.bool_
CAP_VAL_MASK   = IMG_VAL_MASK[cap_gt_img_idx]     # np.bool_

# Sanity: nessuna immagine in comune tra train e val
overlap_imgs = (IMG_TRAIN_MASK & IMG_VAL_MASK).any()
assert not overlap_imgs, f"Overlap immagini train/val > 0"

# ===========================
# 3) Costruisci tensori train/val
# ===========================
X_train = X[CAP_TRAIN_MASK]
y_train = y[CAP_TRAIN_MASK]
X_val   = X[CAP_VAL_MASK]
y_val   = y[CAP_VAL_MASK]

print(f"Train captions: {len(X_train):,} | Val captions: {len(X_val):,}")
print(f"Train images  : {int(IMG_TRAIN_MASK.sum()):,} | Val images  : {int(IMG_VAL_MASK.sum()):,}")

# ===========================
# 4) DataLoader
# ===========================
train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val,   y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=(DEVICE.type=='cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=='cuda'))

# ===========================
# 5) Istanziazione modello
# ===========================
model = SetPredictionHead(
    d_text=X.shape[1],
    d_vis=y.shape[1],
    K=K_SLOTS,
    hidden=2048,
    slot_dropout_p=SLOT_DROPOUT_P
).to(DEVICE)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\n[Train] MaxMatch + Sinkhorn + ISDL + GDL (curriculum su τ) ...")

# ============================================================
# 4) GDL – Global Discriminative Loss (Alomari 2025)
#     Rafforza separazione vs negativi di batch (ranking)
# ============================================================
def gdl_global_discriminative_true(S_T, t_vis_norm, s: float, delta2: float):
    """
    GDL: penalizza allineamento slot ↔ globale (stessa modalità/spazio),
    spingendo gli slot a non collassare sul globale.
    """
    # sim per-slot col globale: (B,K)
    sims = torch.einsum('bkd,bd->bk', S_T, t_vis_norm).clamp(-1,1)
    return torch.exp(s * (sims - delta2)).mean()



# ============================================================
# 5) Diagnostica: log-varianza intra-slot
# ============================================================
@torch.no_grad()
def slot_log_variance(S_T: torch.Tensor) -> float:
    var_fd = S_T.var(dim=1, unbiased=False)   # (B, D)
    var_mean = var_fd.mean(dim=1).mean().clamp_min(1e-8)
    return float(torch.log(var_mean).item())


@torch.no_grad()
def mean_offdiag_cos(E_Tn):
    K = E_Tn.size(1)
    C = torch.matmul(E_Tn, E_Tn.transpose(-1,-2))          # (B,K,K)
    off = C - torch.eye(K, device=E_Tn.device).unsqueeze(0)
    denom = max(K*(K-1), 1)
    return float(off.abs().sum(dim=(1,2)).mean().item() / denom)

def gaussian_kernel(x, y, sigma=1.0):
    x2 = (x*x).sum(dim=1, keepdim=True)
    y2 = (y*y).sum(dim=1, keepdim=True)
    xy = x @ y.t()
    dist = x2 - 2*xy + y2.t()
    return torch.exp(-dist / (2*sigma**2))

def mmd_rbf(x, y, sigma=1.0):
    Kxx = gaussian_kernel(x, x, sigma).mean()
    Kyy = gaussian_kernel(y, y, sigma).mean()
    Kxy = gaussian_kernel(x, y, sigma).mean()
    return Kxx + Kyy - 2*Kxy

def diversity_regularizer_exp(E: torch.Tensor, s: float = 1.0):
    """
    Accetta:
      - E: (B,K,D)  -> viene flattenato a (B*K, D)
      - E: (N,D)    -> usato così com'è
    """
    if E.dim() == 3:
        B, K, D = E.shape
        E = E.reshape(B * K, D)
    elif E.dim() == 2:
        D = E.size(1)
    else:
        raise ValueError(f"diversity_regularizer_exp: atteso 2D/3D, trovato {tuple(E.shape)}")

    C = (E @ E.t()).clamp(-1, 1)  # (N,N)
    mask = ~torch.eye(E.size(0), device=E.device, dtype=torch.bool)
    C_off = C[mask]
    return torch.exp(-2.0 * (1 - C_off)).mean()


def subsample_rows(X: torch.Tensor, max_n: int):
    n = X.size(0)
    if n <= max_n: return X
    idx = torch.randint(0, n, (max_n,), device=X.device)
    return X.index_select(0, idx)

# Subsample per evitare O(n^2) pieno
AGG_MAX = 256      # righe per MMD
RES_MAX = 512      # righe per DIV (BK ~ B*K)
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                epochs: int,
                lr: float,
                logvar_warm_epochs: int = 3) -> nn.Module:

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        logvar_vals, offdiag_vals = [], []
        frac = (epoch - 1) / max(epochs - 1, 1)
        SINKHORN_TAU = SINKHORN_TAU_START + frac * (SINKHORN_TAU_END - SINKHORN_TAU_START)
        SINKHORN_TAU = max(SINKHORN_TAU_END, SINKHORN_TAU)  # floor = *_END
        for Xb, Yb in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}"):


            Xb = Xb.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # --- forward ---
                S_T, E_T, t_vis_n = model(Xb)      # S_T: scoring; E_T: residui (ISDL); t_vis_n: globale normalizzato
                E_Tn = F.normalize(E_T, dim=-1)

                # --- diagnostica ---
                if epoch <= logvar_warm_epochs:
                    logvar_vals.append(slot_log_variance(S_T))
                offdiag_vals.append(mean_offdiag_cos(E_Tn))

                # --- s_H (positivo) per logging: forma chiusa (rapida) ---
                # s_pos = s_h(
                #     S_T, Yb,
                #     tau=SINKHORN_TAU, iters=SINKHORN_ITERS, detach=DETACH_ASSIGNMENT
                # )  # (B,)

                # --- loss principali ---
                # 1) Triplet su S_H (hardest-neg su batch) – usa s_h "wrapper" all'interno
                loss_tri = triplet_maxmatch_sh(
                    S_T, Yb, DELTA_1,
                    tau=SINKHORN_TAU, iters=SINKHORN_ITERS, detach=DETACH_ASSIGNMENT
                )

                # 2) ISDL (intra-set, exp con margine)
                loss_isdl = isdl_intra_set_diversity_exp(E_Tn, s=SCALE_S, delta3=DELTA_3
                )

                # 3) GDL (slot vs globale t_vis nello stesso spazio)
                loss_gdl = gdl_global_discriminative_true(
                                        S_T, t_vis_n, s=SCALE_S, delta2=DELTA_2
                )

                # 4) Opzionali: MMD (media slot ↔ target) e Diversità residui con subsample
                agg_mean = F.normalize(S_T.mean(dim=1), dim=-1)  # (B,D)
                agg_mean_ss = subsample_rows(agg_mean, AGG_MAX)
                Yb_ss       = subsample_rows(F.normalize(Yb, dim=-1).detach(), AGG_MAX)
                loss_mmd = mmd_rbf(agg_mean_ss, Yb_ss, sigma=1.0)

                E_flat_ss = subsample_rows(E_Tn.reshape(-1, E_Tn.size(-1)), RES_MAX)
                loss_div  = diversity_regularizer_exp(E_flat_ss, s=1.0)

                loss = (loss_tri
                        + LAMBDA_ISDL * loss_isdl
                        + LAMBDA_GDL  * loss_gdl
                        + LAMBDA_MMD * loss_mmd
                        + LAMBDA_DIV * loss_div)

            # --- backward + step (AMP) ---
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # --- logging train ---


        # ===== Validation: S_H medio come surrogato, versione rapida =====
        model.eval()
        val_s_h_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for Xb, Yb in DataLoader(val_loader.dataset, batch_size=BATCH_SIZE, shuffle=False):
                Xb = Xb.to(device, non_blocking=True)
                Yb = Yb.to(device, non_blocking=True)
                S_T, _, _ = model(Xb)
                s_h_val = s_h(
                    S_T, Yb,
                    tau=SINKHORN_TAU, iters=SINKHORN_ITERS, detach=DETACH_ASSIGNMENT
                )
                val_s_h_sum += float(s_h_val.mean().item())
                val_batches += 1

        val_s_h_avg = val_s_h_sum / max(val_batches, 1)

        logvar_text = ""
        if len(logvar_vals) > 0:
            logvar_epoch = sum(logvar_vals) / len(logvar_vals)
            logvar_text = f" | log-var(S_T): {logvar_epoch:.2f}"

        print(
            f"val s_score={val_s_h_avg:.4f} | loss_tri={float(loss_tri.item()):.4f} "
            f"| isdl={float(loss_isdl.item()):.4f} "
            f"| gdl={float(loss_gdl.item()):.4f} "
            
            f"| last_loss={float(loss.item()):.4f} |"
            f"offdiag(E_T): {sum(offdiag_vals)/len(offdiag_vals):.4f}{logvar_text}|"
        )

        # --- salva anche checkpoint per epoca ---
        ckpt_dir = Path(MODEL_PATH).parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # --- best model ---
        if val_s_h_avg > best_val:
            best_val = val_s_h_avg
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best (val S_H={val_s_h_avg:.4f}) → {ckpt_path.name}")
        else:
            print(f"  ☐ Saved {ckpt_path.name}")
        sched.step(epoch)
    return model


# ===========================
# 6) Training
# ===========================
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LR
)
```

    (125000,)
    Train data: 125000 captions, 125000 images
    Train captions: 112,420 | Val captions: 12,580
    Train images  : 22,484 | Val images  : 2,516
    Parameters: 14,714,374
    
    [Train] MaxMatch + Sinkhorn + ISDL + GDL (curriculum su τ) ...
    

    C:\Users\lucam\AppData\Local\Temp\ipykernel_27316\3786990716.py:389: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    [Train] Epoch 1/120:   0%|          | 0/440 [00:00<?, ?it/s]C:\Users\lucam\AppData\Local\Temp\ipykernel_27316\3786990716.py:406: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
    [Train] Epoch 1/120: 100%|██████████| 440/440 [00:39<00:00, 11.15it/s]
    

    val s_score=0.3006 | loss_tri=0.3188 | isdl=0.7235 | gdl=1.0495 | last_loss=0.4927 |offdiag(E_T): 0.1043 | log-var(S_T): -8.13|
      ✓ Saved best (val S_H=0.3006) → epoch_001.pth
    

    [Train] Epoch 2/120: 100%|██████████| 440/440 [00:38<00:00, 11.50it/s]
    

    val s_score=0.2950 | loss_tri=0.3176 | isdl=0.7192 | gdl=1.0524 | last_loss=0.4915 |offdiag(E_T): 0.1341 | log-var(S_T): -8.11|
      ☐ Saved epoch_002.pth
    

    [Train] Epoch 3/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2802 | loss_tri=0.3149 | isdl=0.7165 | gdl=1.0542 | last_loss=0.4889 |offdiag(E_T): 0.1406 | log-var(S_T): -8.12|
      ☐ Saved epoch_003.pth
    

    [Train] Epoch 4/120: 100%|██████████| 440/440 [00:38<00:00, 11.54it/s]
    

    val s_score=0.2697 | loss_tri=0.2832 | isdl=0.7188 | gdl=1.0587 | last_loss=0.4578 |offdiag(E_T): 0.1389|
      ☐ Saved epoch_004.pth
    

    [Train] Epoch 5/120: 100%|██████████| 440/440 [00:38<00:00, 11.53it/s]
    

    val s_score=0.2682 | loss_tri=0.2778 | isdl=0.7228 | gdl=1.0656 | last_loss=0.4536 |offdiag(E_T): 0.1301|
      ☐ Saved epoch_005.pth
    

    [Train] Epoch 6/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2588 | loss_tri=0.2815 | isdl=0.7246 | gdl=1.0677 | last_loss=0.4578 |offdiag(E_T): 0.1214|
      ☐ Saved epoch_006.pth
    

    [Train] Epoch 7/120: 100%|██████████| 440/440 [00:38<00:00, 11.50it/s]
    

    val s_score=0.2559 | loss_tri=0.2415 | isdl=0.7297 | gdl=1.0755 | last_loss=0.4194 |offdiag(E_T): 0.1140|
      ☐ Saved epoch_007.pth
    

    [Train] Epoch 8/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2507 | loss_tri=0.2924 | isdl=0.7260 | gdl=1.0705 | last_loss=0.4690 |offdiag(E_T): 0.1088|
      ☐ Saved epoch_008.pth
    

    [Train] Epoch 9/120: 100%|██████████| 440/440 [00:38<00:00, 11.56it/s]
    

    val s_score=0.2505 | loss_tri=0.2635 | isdl=0.7346 | gdl=1.0807 | last_loss=0.4426 |offdiag(E_T): 0.1047|
      ☐ Saved epoch_009.pth
    

    [Train] Epoch 10/120: 100%|██████████| 440/440 [00:38<00:00, 11.55it/s]
    

    val s_score=0.2482 | loss_tri=0.2495 | isdl=0.7307 | gdl=1.0786 | last_loss=0.4276 |offdiag(E_T): 0.1026|
      ☐ Saved epoch_010.pth
    

    [Train] Epoch 11/120: 100%|██████████| 440/440 [00:38<00:00, 11.58it/s]
    

    val s_score=0.2565 | loss_tri=0.2900 | isdl=0.7239 | gdl=1.0663 | last_loss=0.4659 |offdiag(E_T): 0.1106|
      ☐ Saved epoch_011.pth
    

    [Train] Epoch 12/120: 100%|██████████| 440/440 [00:37<00:00, 11.68it/s]
    

    val s_score=0.2576 | loss_tri=0.2677 | isdl=0.7245 | gdl=1.0685 | last_loss=0.4440 |offdiag(E_T): 0.1140|
      ☐ Saved epoch_012.pth
    

    [Train] Epoch 13/120: 100%|██████████| 440/440 [00:37<00:00, 11.65it/s]
    

    val s_score=0.2540 | loss_tri=0.2652 | isdl=0.7285 | gdl=1.0731 | last_loss=0.4424 |offdiag(E_T): 0.1121|
      ☐ Saved epoch_013.pth
    

    [Train] Epoch 14/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2498 | loss_tri=0.2535 | isdl=0.7328 | gdl=1.0771 | last_loss=0.4317 |offdiag(E_T): 0.1090|
      ☐ Saved epoch_014.pth
    

    [Train] Epoch 15/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2483 | loss_tri=0.2519 | isdl=0.7299 | gdl=1.0747 | last_loss=0.4296 |offdiag(E_T): 0.1053|
      ☐ Saved epoch_015.pth
    

    [Train] Epoch 16/120: 100%|██████████| 440/440 [00:37<00:00, 11.64it/s]
    

    val s_score=0.2462 | loss_tri=0.2685 | isdl=0.7328 | gdl=1.0761 | last_loss=0.4466 |offdiag(E_T): 0.1011|
      ☐ Saved epoch_016.pth
    

    [Train] Epoch 17/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2475 | loss_tri=0.2585 | isdl=0.7279 | gdl=1.0711 | last_loss=0.4353 |offdiag(E_T): 0.0968|
      ☐ Saved epoch_017.pth
    

    [Train] Epoch 18/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2443 | loss_tri=0.2504 | isdl=0.7368 | gdl=1.0805 | last_loss=0.4295 |offdiag(E_T): 0.0923|
      ☐ Saved epoch_018.pth
    

    [Train] Epoch 19/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2398 | loss_tri=0.2572 | isdl=0.7378 | gdl=1.0795 | last_loss=0.4361 |offdiag(E_T): 0.0874|
      ☐ Saved epoch_019.pth
    

    [Train] Epoch 20/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2426 | loss_tri=0.2343 | isdl=0.7412 | gdl=1.0828 | last_loss=0.4140 |offdiag(E_T): 0.0822|
      ☐ Saved epoch_020.pth
    

    [Train] Epoch 21/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2414 | loss_tri=0.2209 | isdl=0.7456 | gdl=1.0856 | last_loss=0.4014 |offdiag(E_T): 0.0772|
      ☐ Saved epoch_021.pth
    

    [Train] Epoch 22/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2386 | loss_tri=0.2290 | isdl=0.7436 | gdl=1.0821 | last_loss=0.4088 |offdiag(E_T): 0.0726|
      ☐ Saved epoch_022.pth
    

    [Train] Epoch 23/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2399 | loss_tri=0.1993 | isdl=0.7548 | gdl=1.0900 | last_loss=0.3812 |offdiag(E_T): 0.0683|
      ☐ Saved epoch_023.pth
    

    [Train] Epoch 24/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2399 | loss_tri=0.2400 | isdl=0.7470 | gdl=1.0858 | last_loss=0.4206 |offdiag(E_T): 0.0651|
      ☐ Saved epoch_024.pth
    

    [Train] Epoch 25/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2357 | loss_tri=0.2127 | isdl=0.7531 | gdl=1.0891 | last_loss=0.3943 |offdiag(E_T): 0.0624|
      ☐ Saved epoch_025.pth
    

    [Train] Epoch 26/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2373 | loss_tri=0.2420 | isdl=0.7502 | gdl=1.0862 | last_loss=0.4229 |offdiag(E_T): 0.0604|
      ☐ Saved epoch_026.pth
    

    [Train] Epoch 27/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2373 | loss_tri=0.2166 | isdl=0.7526 | gdl=1.0897 | last_loss=0.3983 |offdiag(E_T): 0.0591|
      ☐ Saved epoch_027.pth
    

    [Train] Epoch 28/120: 100%|██████████| 440/440 [00:38<00:00, 11.58it/s]
    

    val s_score=0.2373 | loss_tri=0.2379 | isdl=0.7476 | gdl=1.0875 | last_loss=0.4188 |offdiag(E_T): 0.0582|
      ☐ Saved epoch_028.pth
    

    [Train] Epoch 29/120: 100%|██████████| 440/440 [00:38<00:00, 11.54it/s]
    

    val s_score=0.2364 | loss_tri=0.2244 | isdl=0.7535 | gdl=1.0890 | last_loss=0.4060 |offdiag(E_T): 0.0576|
      ☐ Saved epoch_029.pth
    

    [Train] Epoch 30/120: 100%|██████████| 440/440 [00:38<00:00, 11.54it/s]
    

    val s_score=0.2353 | loss_tri=0.2231 | isdl=0.7505 | gdl=1.0869 | last_loss=0.4040 |offdiag(E_T): 0.0573|
      ☐ Saved epoch_030.pth
    

    [Train] Epoch 31/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2393 | loss_tri=0.2290 | isdl=0.7437 | gdl=1.0816 | last_loss=0.4088 |offdiag(E_T): 0.0650|
      ☐ Saved epoch_031.pth
    

    [Train] Epoch 32/120: 100%|██████████| 440/440 [00:37<00:00, 11.65it/s]
    

    val s_score=0.2417 | loss_tri=0.2209 | isdl=0.7499 | gdl=1.0845 | last_loss=0.4018 |offdiag(E_T): 0.0704|
      ☐ Saved epoch_032.pth
    

    [Train] Epoch 33/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2431 | loss_tri=0.2198 | isdl=0.7466 | gdl=1.0838 | last_loss=0.4001 |offdiag(E_T): 0.0713|
      ☐ Saved epoch_033.pth
    

    [Train] Epoch 34/120: 100%|██████████| 440/440 [00:37<00:00, 11.65it/s]
    

    val s_score=0.2366 | loss_tri=0.2315 | isdl=0.7460 | gdl=1.0829 | last_loss=0.4117 |offdiag(E_T): 0.0705|
      ☐ Saved epoch_034.pth
    

    [Train] Epoch 35/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2417 | loss_tri=0.2489 | isdl=0.7438 | gdl=1.0816 | last_loss=0.4287 |offdiag(E_T): 0.0693|
      ☐ Saved epoch_035.pth
    

    [Train] Epoch 36/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2380 | loss_tri=0.2509 | isdl=0.7466 | gdl=1.0827 | last_loss=0.4314 |offdiag(E_T): 0.0681|
      ☐ Saved epoch_036.pth
    

    [Train] Epoch 37/120: 100%|██████████| 440/440 [00:38<00:00, 11.48it/s]
    

    val s_score=0.2356 | loss_tri=0.2175 | isdl=0.7493 | gdl=1.0847 | last_loss=0.3981 |offdiag(E_T): 0.0668|
      ☐ Saved epoch_037.pth
    

    [Train] Epoch 38/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2374 | loss_tri=0.2335 | isdl=0.7457 | gdl=1.0832 | last_loss=0.4138 |offdiag(E_T): 0.0654|
      ☐ Saved epoch_038.pth
    

    [Train] Epoch 39/120: 100%|██████████| 440/440 [00:37<00:00, 11.65it/s]
    

    val s_score=0.2344 | loss_tri=0.2510 | isdl=0.7448 | gdl=1.0812 | last_loss=0.4308 |offdiag(E_T): 0.0635|
      ☐ Saved epoch_039.pth
    

    [Train] Epoch 40/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2369 | loss_tri=0.2183 | isdl=0.7530 | gdl=1.0879 | last_loss=0.3999 |offdiag(E_T): 0.0621|
      ☐ Saved epoch_040.pth
    

    [Train] Epoch 41/120: 100%|██████████| 440/440 [00:37<00:00, 11.66it/s]
    

    val s_score=0.2369 | loss_tri=0.2427 | isdl=0.7493 | gdl=1.0839 | last_loss=0.4234 |offdiag(E_T): 0.0607|
      ☐ Saved epoch_041.pth
    

    [Train] Epoch 42/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2375 | loss_tri=0.2185 | isdl=0.7530 | gdl=1.0867 | last_loss=0.3998 |offdiag(E_T): 0.0596|
      ☐ Saved epoch_042.pth
    

    [Train] Epoch 43/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2393 | loss_tri=0.2221 | isdl=0.7520 | gdl=1.0860 | last_loss=0.4032 |offdiag(E_T): 0.0583|
      ☐ Saved epoch_043.pth
    

    [Train] Epoch 44/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2393 | loss_tri=0.2260 | isdl=0.7565 | gdl=1.0888 | last_loss=0.4080 |offdiag(E_T): 0.0575|
      ☐ Saved epoch_044.pth
    

    [Train] Epoch 45/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2370 | loss_tri=0.2161 | isdl=0.7563 | gdl=1.0878 | last_loss=0.3980 |offdiag(E_T): 0.0567|
      ☐ Saved epoch_045.pth
    

    [Train] Epoch 46/120: 100%|██████████| 440/440 [00:37<00:00, 11.66it/s]
    

    val s_score=0.2349 | loss_tri=0.2215 | isdl=0.7588 | gdl=1.0897 | last_loss=0.4038 |offdiag(E_T): 0.0560|
      ☐ Saved epoch_046.pth
    

    [Train] Epoch 47/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2356 | loss_tri=0.2110 | isdl=0.7672 | gdl=1.0928 | last_loss=0.3943 |offdiag(E_T): 0.0559|
      ☐ Saved epoch_047.pth
    

    [Train] Epoch 48/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2321 | loss_tri=0.2072 | isdl=0.7593 | gdl=1.0897 | last_loss=0.3896 |offdiag(E_T): 0.0557|
      ☐ Saved epoch_048.pth
    

    [Train] Epoch 49/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2329 | loss_tri=0.2043 | isdl=0.7596 | gdl=1.0899 | last_loss=0.3867 |offdiag(E_T): 0.0557|
      ☐ Saved epoch_049.pth
    

    [Train] Epoch 50/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2345 | loss_tri=0.2087 | isdl=0.7636 | gdl=1.0917 | last_loss=0.3917 |offdiag(E_T): 0.0561|
      ☐ Saved epoch_050.pth
    

    [Train] Epoch 51/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2340 | loss_tri=0.2028 | isdl=0.7637 | gdl=1.0909 | last_loss=0.3857 |offdiag(E_T): 0.0566|
      ☐ Saved epoch_051.pth
    

    [Train] Epoch 52/120: 100%|██████████| 440/440 [00:37<00:00, 11.64it/s]
    

    val s_score=0.2337 | loss_tri=0.2051 | isdl=0.7729 | gdl=1.0948 | last_loss=0.3896 |offdiag(E_T): 0.0573|
      ☐ Saved epoch_052.pth
    

    [Train] Epoch 53/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2345 | loss_tri=0.1970 | isdl=0.7758 | gdl=1.0967 | last_loss=0.3819 |offdiag(E_T): 0.0583|
      ☐ Saved epoch_053.pth
    

    [Train] Epoch 54/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2293 | loss_tri=0.2102 | isdl=0.7694 | gdl=1.0935 | last_loss=0.3940 |offdiag(E_T): 0.0592|
      ☐ Saved epoch_054.pth
    

    [Train] Epoch 55/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2342 | loss_tri=0.2113 | isdl=0.7714 | gdl=1.0950 | last_loss=0.3955 |offdiag(E_T): 0.0602|
      ☐ Saved epoch_055.pth
    

    [Train] Epoch 56/120: 100%|██████████| 440/440 [00:37<00:00, 11.65it/s]
    

    val s_score=0.2350 | loss_tri=0.1826 | isdl=0.7738 | gdl=1.0956 | last_loss=0.3672 |offdiag(E_T): 0.0615|
      ☐ Saved epoch_056.pth
    

    [Train] Epoch 57/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2338 | loss_tri=0.1950 | isdl=0.7680 | gdl=1.0929 | last_loss=0.3785 |offdiag(E_T): 0.0628|
      ☐ Saved epoch_057.pth
    

    [Train] Epoch 58/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2320 | loss_tri=0.1873 | isdl=0.7759 | gdl=1.0969 | last_loss=0.3721 |offdiag(E_T): 0.0639|
      ☐ Saved epoch_058.pth
    

    [Train] Epoch 59/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2312 | loss_tri=0.1767 | isdl=0.7759 | gdl=1.0979 | last_loss=0.3615 |offdiag(E_T): 0.0651|
      ☐ Saved epoch_059.pth
    

    [Train] Epoch 60/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2317 | loss_tri=0.1979 | isdl=0.7757 | gdl=1.0978 | last_loss=0.3827 |offdiag(E_T): 0.0661|
      ☐ Saved epoch_060.pth
    

    [Train] Epoch 61/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2315 | loss_tri=0.1996 | isdl=0.7775 | gdl=1.0983 | last_loss=0.3848 |offdiag(E_T): 0.0673|
      ☐ Saved epoch_061.pth
    

    [Train] Epoch 62/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2311 | loss_tri=0.1824 | isdl=0.7766 | gdl=1.0958 | last_loss=0.3671 |offdiag(E_T): 0.0686|
      ☐ Saved epoch_062.pth
    

    [Train] Epoch 63/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2315 | loss_tri=0.2024 | isdl=0.7856 | gdl=1.1015 | last_loss=0.3887 |offdiag(E_T): 0.0695|
      ☐ Saved epoch_063.pth
    

    [Train] Epoch 64/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2301 | loss_tri=0.1984 | isdl=0.7778 | gdl=1.0982 | last_loss=0.3835 |offdiag(E_T): 0.0702|
      ☐ Saved epoch_064.pth
    

    [Train] Epoch 65/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2300 | loss_tri=0.1930 | isdl=0.7746 | gdl=1.0965 | last_loss=0.3777 |offdiag(E_T): 0.0711|
      ☐ Saved epoch_065.pth
    

    [Train] Epoch 66/120: 100%|██████████| 440/440 [00:37<00:00, 11.65it/s]
    

    val s_score=0.2293 | loss_tri=0.1857 | isdl=0.7844 | gdl=1.0998 | last_loss=0.3717 |offdiag(E_T): 0.0716|
      ☐ Saved epoch_066.pth
    

    [Train] Epoch 67/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2297 | loss_tri=0.1699 | isdl=0.7861 | gdl=1.1014 | last_loss=0.3562 |offdiag(E_T): 0.0720|
      ☐ Saved epoch_067.pth
    

    [Train] Epoch 68/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2310 | loss_tri=0.2016 | isdl=0.7778 | gdl=1.0987 | last_loss=0.3870 |offdiag(E_T): 0.0723|
      ☐ Saved epoch_068.pth
    

    [Train] Epoch 69/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2306 | loss_tri=0.1912 | isdl=0.7904 | gdl=1.1024 | last_loss=0.3780 |offdiag(E_T): 0.0725|
      ☐ Saved epoch_069.pth
    

    [Train] Epoch 70/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2303 | loss_tri=0.1729 | isdl=0.7837 | gdl=1.1002 | last_loss=0.3589 |offdiag(E_T): 0.0726|
      ☐ Saved epoch_070.pth
    

    [Train] Epoch 71/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2322 | loss_tri=0.2238 | isdl=0.7595 | gdl=1.0868 | last_loss=0.4059 |offdiag(E_T): 0.0669|
      ☐ Saved epoch_071.pth
    

    [Train] Epoch 72/120: 100%|██████████| 440/440 [00:38<00:00, 11.55it/s]
    

    val s_score=0.2342 | loss_tri=0.2132 | isdl=0.7635 | gdl=1.0913 | last_loss=0.3963 |offdiag(E_T): 0.0580|
      ☐ Saved epoch_072.pth
    

    [Train] Epoch 73/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2378 | loss_tri=0.2131 | isdl=0.7686 | gdl=1.0925 | last_loss=0.3967 |offdiag(E_T): 0.0569|
      ☐ Saved epoch_073.pth
    

    [Train] Epoch 74/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2365 | loss_tri=0.2197 | isdl=0.7629 | gdl=1.0917 | last_loss=0.4025 |offdiag(E_T): 0.0567|
      ☐ Saved epoch_074.pth
    

    [Train] Epoch 75/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2339 | loss_tri=0.1994 | isdl=0.7674 | gdl=1.0927 | last_loss=0.3829 |offdiag(E_T): 0.0564|
      ☐ Saved epoch_075.pth
    

    [Train] Epoch 76/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2377 | loss_tri=0.2215 | isdl=0.7636 | gdl=1.0896 | last_loss=0.4041 |offdiag(E_T): 0.0561|
      ☐ Saved epoch_076.pth
    

    [Train] Epoch 77/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2374 | loss_tri=0.2207 | isdl=0.7603 | gdl=1.0882 | last_loss=0.4030 |offdiag(E_T): 0.0561|
      ☐ Saved epoch_077.pth
    

    [Train] Epoch 78/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2400 | loss_tri=0.1918 | isdl=0.7637 | gdl=1.0907 | last_loss=0.3746 |offdiag(E_T): 0.0562|
      ☐ Saved epoch_078.pth
    

    [Train] Epoch 79/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2373 | loss_tri=0.2137 | isdl=0.7677 | gdl=1.0919 | last_loss=0.3971 |offdiag(E_T): 0.0566|
      ☐ Saved epoch_079.pth
    

    [Train] Epoch 80/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2313 | loss_tri=0.2091 | isdl=0.7634 | gdl=1.0916 | last_loss=0.3922 |offdiag(E_T): 0.0566|
      ☐ Saved epoch_080.pth
    

    [Train] Epoch 81/120: 100%|██████████| 440/440 [00:38<00:00, 11.56it/s]
    

    val s_score=0.2386 | loss_tri=0.2051 | isdl=0.7635 | gdl=1.0907 | last_loss=0.3880 |offdiag(E_T): 0.0569|
      ☐ Saved epoch_081.pth
    

    [Train] Epoch 82/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2366 | loss_tri=0.1914 | isdl=0.7696 | gdl=1.0927 | last_loss=0.3751 |offdiag(E_T): 0.0570|
      ☐ Saved epoch_082.pth
    

    [Train] Epoch 83/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2329 | loss_tri=0.2129 | isdl=0.7696 | gdl=1.0914 | last_loss=0.3965 |offdiag(E_T): 0.0575|
      ☐ Saved epoch_083.pth
    

    [Train] Epoch 84/120: 100%|██████████| 440/440 [00:38<00:00, 11.56it/s]
    

    val s_score=0.2313 | loss_tri=0.2125 | isdl=0.7614 | gdl=1.0886 | last_loss=0.3949 |offdiag(E_T): 0.0580|
      ☐ Saved epoch_084.pth
    

    [Train] Epoch 85/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2331 | loss_tri=0.2105 | isdl=0.7604 | gdl=1.0892 | last_loss=0.3933 |offdiag(E_T): 0.0586|
      ☐ Saved epoch_085.pth
    

    [Train] Epoch 86/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2303 | loss_tri=0.2078 | isdl=0.7639 | gdl=1.0917 | last_loss=0.3908 |offdiag(E_T): 0.0588|
      ☐ Saved epoch_086.pth
    

    [Train] Epoch 87/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2323 | loss_tri=0.2293 | isdl=0.7659 | gdl=1.0922 | last_loss=0.4126 |offdiag(E_T): 0.0594|
      ☐ Saved epoch_087.pth
    

    [Train] Epoch 88/120: 100%|██████████| 440/440 [00:38<00:00, 11.52it/s]
    

    val s_score=0.2346 | loss_tri=0.1926 | isdl=0.7755 | gdl=1.0957 | last_loss=0.3773 |offdiag(E_T): 0.0599|
      ☐ Saved epoch_088.pth
    

    [Train] Epoch 89/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2346 | loss_tri=0.2317 | isdl=0.7678 | gdl=1.0921 | last_loss=0.4152 |offdiag(E_T): 0.0610|
      ☐ Saved epoch_089.pth
    

    [Train] Epoch 90/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2336 | loss_tri=0.2035 | isdl=0.7667 | gdl=1.0915 | last_loss=0.3870 |offdiag(E_T): 0.0615|
      ☐ Saved epoch_090.pth
    

    [Train] Epoch 91/120: 100%|██████████| 440/440 [00:38<00:00, 11.58it/s]
    

    val s_score=0.2307 | loss_tri=0.1908 | isdl=0.7690 | gdl=1.0929 | last_loss=0.3746 |offdiag(E_T): 0.0621|
      ☐ Saved epoch_091.pth
    

    [Train] Epoch 92/120: 100%|██████████| 440/440 [00:38<00:00, 11.56it/s]
    

    val s_score=0.2308 | loss_tri=0.1882 | isdl=0.7712 | gdl=1.0945 | last_loss=0.3724 |offdiag(E_T): 0.0628|
      ☐ Saved epoch_092.pth
    

    [Train] Epoch 93/120: 100%|██████████| 440/440 [00:37<00:00, 11.62it/s]
    

    val s_score=0.2294 | loss_tri=0.2074 | isdl=0.7804 | gdl=1.0976 | last_loss=0.3930 |offdiag(E_T): 0.0637|
      ☐ Saved epoch_093.pth
    

    [Train] Epoch 94/120: 100%|██████████| 440/440 [00:38<00:00, 11.53it/s]
    

    val s_score=0.2302 | loss_tri=0.1864 | isdl=0.7747 | gdl=1.0947 | last_loss=0.3709 |offdiag(E_T): 0.0642|
      ☐ Saved epoch_094.pth
    

    [Train] Epoch 95/120: 100%|██████████| 440/440 [00:38<00:00, 11.37it/s]
    

    val s_score=0.2321 | loss_tri=0.2064 | isdl=0.7703 | gdl=1.0937 | last_loss=0.3904 |offdiag(E_T): 0.0652|
      ☐ Saved epoch_095.pth
    

    [Train] Epoch 96/120: 100%|██████████| 440/440 [00:38<00:00, 11.55it/s]
    

    val s_score=0.2332 | loss_tri=0.1856 | isdl=0.7806 | gdl=1.0974 | last_loss=0.3711 |offdiag(E_T): 0.0661|
      ☐ Saved epoch_096.pth
    

    [Train] Epoch 97/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2307 | loss_tri=0.2114 | isdl=0.7711 | gdl=1.0943 | last_loss=0.3955 |offdiag(E_T): 0.0672|
      ☐ Saved epoch_097.pth
    

    [Train] Epoch 98/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2328 | loss_tri=0.1899 | isdl=0.7697 | gdl=1.0930 | last_loss=0.3735 |offdiag(E_T): 0.0681|
      ☐ Saved epoch_098.pth
    

    [Train] Epoch 99/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2357 | loss_tri=0.2091 | isdl=0.7682 | gdl=1.0922 | last_loss=0.3924 |offdiag(E_T): 0.0691|
      ☐ Saved epoch_099.pth
    

    [Train] Epoch 100/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2267 | loss_tri=0.2042 | isdl=0.7796 | gdl=1.0969 | last_loss=0.3896 |offdiag(E_T): 0.0698|
      ☐ Saved epoch_100.pth
    

    [Train] Epoch 101/120: 100%|██████████| 440/440 [00:38<00:00, 11.52it/s]
    

    val s_score=0.2314 | loss_tri=0.1863 | isdl=0.7865 | gdl=1.1003 | last_loss=0.3725 |offdiag(E_T): 0.0711|
      ☐ Saved epoch_101.pth
    

    [Train] Epoch 102/120: 100%|██████████| 440/440 [00:38<00:00, 11.57it/s]
    

    val s_score=0.2271 | loss_tri=0.1963 | isdl=0.7798 | gdl=1.0969 | last_loss=0.3817 |offdiag(E_T): 0.0720|
      ☐ Saved epoch_102.pth
    

    [Train] Epoch 103/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2307 | loss_tri=0.1923 | isdl=0.7796 | gdl=1.0974 | last_loss=0.3778 |offdiag(E_T): 0.0731|
      ☐ Saved epoch_103.pth
    

    [Train] Epoch 104/120: 100%|██████████| 440/440 [00:38<00:00, 11.56it/s]
    

    val s_score=0.2288 | loss_tri=0.1738 | isdl=0.7868 | gdl=1.0995 | last_loss=0.3603 |offdiag(E_T): 0.0744|
      ☐ Saved epoch_104.pth
    

    [Train] Epoch 105/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2268 | loss_tri=0.2013 | isdl=0.7782 | gdl=1.0969 | last_loss=0.3866 |offdiag(E_T): 0.0755|
      ☐ Saved epoch_105.pth
    

    [Train] Epoch 106/120: 100%|██████████| 440/440 [00:38<00:00, 11.55it/s]
    

    val s_score=0.2315 | loss_tri=0.1750 | isdl=0.7896 | gdl=1.1014 | last_loss=0.3617 |offdiag(E_T): 0.0767|
      ☐ Saved epoch_106.pth
    

    [Train] Epoch 107/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2301 | loss_tri=0.1743 | isdl=0.7919 | gdl=1.1021 | last_loss=0.3613 |offdiag(E_T): 0.0779|
      ☐ Saved epoch_107.pth
    

    [Train] Epoch 108/120: 100%|██████████| 440/440 [00:37<00:00, 11.58it/s]
    

    val s_score=0.2273 | loss_tri=0.1764 | isdl=0.7926 | gdl=1.1019 | last_loss=0.3636 |offdiag(E_T): 0.0791|
      ☐ Saved epoch_108.pth
    

    [Train] Epoch 109/120: 100%|██████████| 440/440 [00:37<00:00, 11.60it/s]
    

    val s_score=0.2286 | loss_tri=0.2076 | isdl=0.7812 | gdl=1.0961 | last_loss=0.3928 |offdiag(E_T): 0.0805|
      ☐ Saved epoch_109.pth
    

    [Train] Epoch 110/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2309 | loss_tri=0.1970 | isdl=0.7858 | gdl=1.0985 | last_loss=0.3832 |offdiag(E_T): 0.0814|
      ☐ Saved epoch_110.pth
    

    [Train] Epoch 111/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2297 | loss_tri=0.1651 | isdl=0.7963 | gdl=1.1041 | last_loss=0.3528 |offdiag(E_T): 0.0824|
      ☐ Saved epoch_111.pth
    

    [Train] Epoch 112/120: 100%|██████████| 440/440 [00:38<00:00, 11.54it/s]
    

    val s_score=0.2305 | loss_tri=0.1718 | isdl=0.7923 | gdl=1.1017 | last_loss=0.3589 |offdiag(E_T): 0.0840|
      ☐ Saved epoch_112.pth
    

    [Train] Epoch 113/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2295 | loss_tri=0.1685 | isdl=0.7908 | gdl=1.1010 | last_loss=0.3554 |offdiag(E_T): 0.0851|
      ☐ Saved epoch_113.pth
    

    [Train] Epoch 114/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2302 | loss_tri=0.1642 | isdl=0.7913 | gdl=1.0999 | last_loss=0.3509 |offdiag(E_T): 0.0864|
      ☐ Saved epoch_114.pth
    

    [Train] Epoch 115/120: 100%|██████████| 440/440 [00:37<00:00, 11.61it/s]
    

    val s_score=0.2288 | loss_tri=0.1919 | isdl=0.7892 | gdl=1.1011 | last_loss=0.3786 |offdiag(E_T): 0.0876|
      ☐ Saved epoch_115.pth
    

    [Train] Epoch 116/120: 100%|██████████| 440/440 [00:37<00:00, 11.64it/s]
    

    val s_score=0.2282 | loss_tri=0.1975 | isdl=0.7882 | gdl=1.1008 | last_loss=0.3842 |offdiag(E_T): 0.0889|
      ☐ Saved epoch_116.pth
    

    [Train] Epoch 117/120: 100%|██████████| 440/440 [00:37<00:00, 11.63it/s]
    

    val s_score=0.2284 | loss_tri=0.1886 | isdl=0.7917 | gdl=1.1022 | last_loss=0.3757 |offdiag(E_T): 0.0899|
      ☐ Saved epoch_117.pth
    

    [Train] Epoch 118/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2279 | loss_tri=0.1532 | isdl=0.7910 | gdl=1.1011 | last_loss=0.3401 |offdiag(E_T): 0.0911|
      ☐ Saved epoch_118.pth
    

    [Train] Epoch 119/120: 100%|██████████| 440/440 [00:37<00:00, 11.59it/s]
    

    val s_score=0.2261 | loss_tri=0.1772 | isdl=0.7949 | gdl=1.1035 | last_loss=0.3648 |offdiag(E_T): 0.0924|
      ☐ Saved epoch_119.pth
    

    [Train] Epoch 120/120: 100%|██████████| 440/440 [00:37<00:00, 11.64it/s]
    

    val s_score=0.2271 | loss_tri=0.1843 | isdl=0.7882 | gdl=1.1001 | last_loss=0.3708 |offdiag(E_T): 0.0936|
      ☐ Saved epoch_120.pth
    


```python
# ===========================
# Valutazione: seleziona il best checkpoint per MRR (val gallery)
# ===========================
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from challenge.src.eval.metrics import mrr, ndcg, recall_at_k  # intoccabili

model.eval()

# ============================================================
# Aggregazione slot -> 1 embedding (submission-compatibile)
# ============================================================
@torch.no_grad()
def aggregate_slots(S_T: torch.Tensor, V_ref: torch.Tensor | None = None, mode: str = "mean"):
    assert S_T.dim() == 3, f"aggregate_slots: atteso (B,K,D), trovato {tuple(S_T.shape)}"
    if mode == "mean" or V_ref is None:
        out = S_T.mean(dim=1)
        return F.normalize(out, dim=-1)
    # winner-slot
    assert V_ref is not None and V_ref.dim() == 2 and V_ref.size(0) == S_T.size(0)
    Vn = F.normalize(V_ref, dim=-1)
    sims = torch.einsum('bkd,bd->bk', S_T, Vn)
    idx  = sims.argmax(dim=1)
    out  = S_T[torch.arange(S_T.size(0), device=S_T.device), idx, :]
    return F.normalize(out, dim=-1)

# ===========================
# Split COERENTE per IMMAGINI (stesso hashing del train)
# ===========================
import hashlib
from scipy import sparse

img_names_all = train_data['images/names']                                  # (N_img,)
img_emb_all   = torch.from_numpy(train_data['images/embeddings']).float()   # (N_img, D_img)

val_ratio = 0.10
def stable_hash(name: str) -> float:
    h = hashlib.md5(name.encode('utf-8')).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

img_hash       = np.array([stable_hash(nm) for nm in img_names_all])
IMG_VAL_MASK   = (img_hash < val_ratio)
IMG_TRAIN_MASK = ~IMG_VAL_MASK
assert not (IMG_VAL_MASK & IMG_TRAIN_MASK).any(), "Overlap immagini train/val > 0"

# Caption → indice globale immagine GT
cap_to_img = train_data['captions/label']
if sparse.issparse(cap_to_img):
    cap_gt_img_idx = cap_to_img.argmax(axis=1).A1
else:
    cap_gt_img_idx = np.argmax(cap_to_img, axis=1)

CAP_TRAIN_MASK = IMG_TRAIN_MASK[cap_gt_img_idx]
CAP_VAL_MASK   = IMG_VAL_MASK[cap_gt_img_idx]

# Tensors val (caption→embedding, image target embedding)
X_val = X[CAP_VAL_MASK]
y_val = y[CAP_VAL_MASK]
val_dataset = TensorDataset(X_val, y_val)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=='cuda'))

print(f"[Eval] Val captions: {len(X_val):,} | Val images: {int(IMG_VAL_MASK.sum()):,}")

# Gallery VALIDAZIONE (solo immagini di val, normalizzate)
val_img_embd = F.normalize(img_emb_all[torch.from_numpy(IMG_VAL_MASK)], dim=-1).cpu()  # (N_img_val, D)
val_img_file = img_names_all[IMG_VAL_MASK]

# Rimappa GT caption→indice compatto della gallery val
global_to_val = -np.ones(len(img_names_all), dtype=np.int64)
global_to_val[np.where(IMG_VAL_MASK)[0]] = np.arange(IMG_VAL_MASK.sum(), dtype=np.int64)
val_gt_global = cap_gt_img_idx[CAP_VAL_MASK]
val_label = global_to_val[val_gt_global]   # (Nq,)

# ============================================================
# Helper: calcolo metriche retrieval su gallery completa
# ============================================================
@torch.no_grad()
def evaluate_retrieval_global(Z: torch.Tensor,
                              gallery: torch.Tensor,
                              gt_indices: np.ndarray,
                              topk: int = 100,
                              chunk: int = 512):
    Z = Z.to('cpu'); gallery = gallery.to('cpu')
    Nq, Ng = Z.size(0), gallery.size(0)
    topk = min(topk, Ng)

    all_topk = []
    for start in range(0, Nq, chunk):
        end = min(start + chunk, Nq)
        sims = Z[start:end] @ gallery.T
        topk_idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True).indices
        all_topk.append(topk_idx.cpu().numpy())
    pred_indices = np.vstack(all_topk).astype(np.int64)

    l2_dist = (Z - gallery[torch.from_numpy(gt_indices)]).norm(dim=1).mean().item()
    return {
        'mrr': mrr(pred_indices, gt_indices),
        'ndcg': ndcg(pred_indices, gt_indices),
        'recall_at_1':  recall_at_k(pred_indices, gt_indices, 1),
        'recall_at_3':  recall_at_k(pred_indices, gt_indices, 3),
        'recall_at_5':  recall_at_k(pred_indices, gt_indices, 5),
        'recall_at_10': recall_at_k(pred_indices, gt_indices, 10),
        'recall_at_50': recall_at_k(pred_indices, gt_indices, 50),
        'l2_dist': l2_dist,
    }

# ============================================================
# Scan dei checkpoint → selezione BEST per MRR (stessa val-gallery)
# ============================================================
ckpt_dir = Path(MODEL_PATH).parent / "checkpoints"
checkpoints = sorted(ckpt_dir.glob("epoch_*.pth"))
assert len(checkpoints) > 0, f"Nessun checkpoint trovato in {ckpt_dir}"

results = []
with torch.no_grad():
    for ckpt_path in checkpoints:
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        preds_val = []
        for Xb, _ in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False):
            Xb = Xb.to(DEVICE, non_blocking=True)
            S_Tb, _, _ = model(Xb)
            Eb = aggregate_slots(S_Tb, mode="mean")
            preds_val.append(Eb.cpu())
        Z_val = F.normalize(torch.cat(preds_val, dim=0), dim=-1).cpu()

        sims = Z_val @ val_img_embd.T
        topk_idx = torch.topk(sims, k=min(100, val_img_embd.size(0)), dim=1, largest=True, sorted=True).indices.numpy()
        mrr_val = mrr(topk_idx, val_label)

        results.append((ckpt_path.name, float(mrr_val)))
        print(f"{ckpt_path.name:15s} → MRR={mrr_val:.5f}")

best_ckpt_name, best_mrr = max(results, key=lambda x: x[1])
print("\n=== Miglior checkpoint (MRR su val-gallery) ===")
print(f"{best_ckpt_name} → MRR={best_mrr:.5f}")

# Carica il best nel modello e stampa le metriche complete (MRR, NDCG, Recall@K, L2)
best_state = torch.load(ckpt_dir / best_ckpt_name, map_location=DEVICE)
model.load_state_dict(best_state)
model.eval()

# Pred embedding val col best
preds_val = []
with torch.no_grad():
    for Xb, _ in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False):
        Xb = Xb.to(DEVICE, non_blocking=True)
        S_Tb, _, _ = model(Xb)
        Eb = aggregate_slots(S_Tb, mode="mean")
        preds_val.append(Eb.cpu())
Z_val_best = F.normalize(torch.cat(preds_val, dim=0), dim=-1).cpu()

res_val = evaluate_retrieval_global(Z_val_best, val_img_embd, val_label, topk=100, chunk=512)
print("\n=== Val (image-level split, global gallery) — BEST CKPT ===")
for k, v in res_val.items():
    print(f"{k:15s}: {v:.4f}")

# (Facoltativo) Esplora retrieval qualitativo su 3 esempi
for _ in range(3):
    i = np.random.randint(0, len(X_val))
    with torch.no_grad():
        Sb, _, _ = model(X_val[i:i+1].to(DEVICE))
        zb = aggregate_slots(Sb, mode="mean").cpu().squeeze(0)
    caption_text = train_data['captions/text'][CAP_VAL_MASK][i]
    gt_idx = int(val_label[i])
    visualize_retrieval(zb, gt_idx, val_img_file, caption_text, val_img_embd, k=5)

# Mantieni in memoria per la submission:
BEST_CHECKPOINT_FOR_SUBMIT = best_ckpt_name
print(f"\n[Ready] BEST_CHECKPOINT_FOR_SUBMIT = {BEST_CHECKPOINT_FOR_SUBMIT}")

```

    [Eval] Val captions: 12,580 | Val images: 2,516
    epoch_001.pth   → MRR=0.42384
    epoch_002.pth   → MRR=0.44404
    epoch_003.pth   → MRR=0.44908
    epoch_004.pth   → MRR=0.45069
    epoch_005.pth   → MRR=0.45406
    epoch_006.pth   → MRR=0.45170
    epoch_007.pth   → MRR=0.44866
    epoch_008.pth   → MRR=0.44671
    epoch_009.pth   → MRR=0.44968
    epoch_010.pth   → MRR=0.44911
    epoch_011.pth   → MRR=0.43893
    epoch_012.pth   → MRR=0.43729
    epoch_013.pth   → MRR=0.43746
    epoch_014.pth   → MRR=0.43673
    epoch_015.pth   → MRR=0.43105
    epoch_016.pth   → MRR=0.43069
    epoch_017.pth   → MRR=0.43152
    epoch_018.pth   → MRR=0.42764
    epoch_019.pth   → MRR=0.42180
    epoch_020.pth   → MRR=0.42272
    epoch_021.pth   → MRR=0.42001
    epoch_022.pth   → MRR=0.41903
    epoch_023.pth   → MRR=0.41535
    epoch_024.pth   → MRR=0.41635
    epoch_025.pth   → MRR=0.41440
    epoch_026.pth   → MRR=0.41347
    epoch_027.pth   → MRR=0.41275
    epoch_028.pth   → MRR=0.41119
    epoch_029.pth   → MRR=0.41155
    epoch_030.pth   → MRR=0.41072
    epoch_031.pth   → MRR=0.40733
    epoch_032.pth   → MRR=0.40860
    epoch_033.pth   → MRR=0.40724
    epoch_034.pth   → MRR=0.40635
    epoch_035.pth   → MRR=0.40681
    epoch_036.pth   → MRR=0.40366
    epoch_037.pth   → MRR=0.40543
    epoch_038.pth   → MRR=0.40338
    epoch_039.pth   → MRR=0.39987
    epoch_040.pth   → MRR=0.40206
    epoch_041.pth   → MRR=0.39931
    epoch_042.pth   → MRR=0.39763
    epoch_043.pth   → MRR=0.39532
    epoch_044.pth   → MRR=0.39322
    epoch_045.pth   → MRR=0.39446
    epoch_046.pth   → MRR=0.39040
    epoch_047.pth   → MRR=0.39351
    epoch_048.pth   → MRR=0.39258
    epoch_049.pth   → MRR=0.38755
    epoch_050.pth   → MRR=0.38907
    epoch_051.pth   → MRR=0.38824
    epoch_052.pth   → MRR=0.38721
    epoch_053.pth   → MRR=0.39063
    epoch_054.pth   → MRR=0.38474
    epoch_055.pth   → MRR=0.38406
    epoch_056.pth   → MRR=0.38450
    epoch_057.pth   → MRR=0.38338
    epoch_058.pth   → MRR=0.38147
    epoch_059.pth   → MRR=0.38064
    epoch_060.pth   → MRR=0.37960
    epoch_061.pth   → MRR=0.37853
    epoch_062.pth   → MRR=0.37896
    epoch_063.pth   → MRR=0.37851
    epoch_064.pth   → MRR=0.37771
    epoch_065.pth   → MRR=0.37747
    epoch_066.pth   → MRR=0.37714
    epoch_067.pth   → MRR=0.37730
    epoch_068.pth   → MRR=0.37736
    epoch_069.pth   → MRR=0.37798
    epoch_070.pth   → MRR=0.37777
    epoch_071.pth   → MRR=0.37748
    epoch_072.pth   → MRR=0.37742
    epoch_073.pth   → MRR=0.38648
    epoch_074.pth   → MRR=0.38154
    epoch_075.pth   → MRR=0.38410
    epoch_076.pth   → MRR=0.38169
    epoch_077.pth   → MRR=0.37989
    epoch_078.pth   → MRR=0.38125
    epoch_079.pth   → MRR=0.38052
    epoch_080.pth   → MRR=0.37858
    epoch_081.pth   → MRR=0.38094
    epoch_082.pth   → MRR=0.37579
    epoch_083.pth   → MRR=0.38039
    epoch_084.pth   → MRR=0.37409
    epoch_085.pth   → MRR=0.37322
    epoch_086.pth   → MRR=0.37719
    epoch_087.pth   → MRR=0.37209
    epoch_088.pth   → MRR=0.37734
    epoch_089.pth   → MRR=0.37265
    epoch_090.pth   → MRR=0.37447
    epoch_091.pth   → MRR=0.37262
    epoch_092.pth   → MRR=0.37163
    epoch_093.pth   → MRR=0.37178
    epoch_094.pth   → MRR=0.37197
    epoch_095.pth   → MRR=0.36949
    epoch_096.pth   → MRR=0.36979
    epoch_097.pth   → MRR=0.37168
    epoch_098.pth   → MRR=0.36792
    epoch_099.pth   → MRR=0.36637
    epoch_100.pth   → MRR=0.36902
    epoch_101.pth   → MRR=0.36747
    epoch_102.pth   → MRR=0.36365
    epoch_103.pth   → MRR=0.36735
    epoch_104.pth   → MRR=0.36407
    epoch_105.pth   → MRR=0.36316
    epoch_106.pth   → MRR=0.36405
    epoch_107.pth   → MRR=0.36106
    epoch_108.pth   → MRR=0.36259
    epoch_109.pth   → MRR=0.36623
    epoch_110.pth   → MRR=0.36571
    epoch_111.pth   → MRR=0.36161
    epoch_112.pth   → MRR=0.35944
    epoch_113.pth   → MRR=0.36125
    epoch_114.pth   → MRR=0.35933
    epoch_115.pth   → MRR=0.35827
    epoch_116.pth   → MRR=0.35743
    epoch_117.pth   → MRR=0.35866
    epoch_118.pth   → MRR=0.35932
    epoch_119.pth   → MRR=0.35749
    epoch_120.pth   → MRR=0.35773
    
    === Miglior checkpoint (MRR su val-gallery) ===
    epoch_005.pth → MRR=0.45406
    
    === Val (image-level split, global gallery) — BEST CKPT ===
    mrr            : 0.4541
    ndcg           : 0.5598
    recall_at_1    : 0.3175
    recall_at_3    : 0.5214
    recall_at_5    : 0.6229
    recall_at_10   : 0.7355
    recall_at_50   : 0.9060
    l2_dist        : 1.1306
    


    
![png](BEST%20SCORE%20CON%20VALUTAZIONE%20GIUSTA%20TEST_files/BEST%20SCORE%20CON%20VALUTAZIONE%20GIUSTA%20TEST_1_1.png)
    



    
![png](BEST%20SCORE%20CON%20VALUTAZIONE%20GIUSTA%20TEST_files/BEST%20SCORE%20CON%20VALUTAZIONE%20GIUSTA%20TEST_1_2.png)
    



    
![png](BEST%20SCORE%20CON%20VALUTAZIONE%20GIUSTA%20TEST_files/BEST%20SCORE%20CON%20VALUTAZIONE%20GIUSTA%20TEST_1_3.png)
    


    
    [Ready] BEST_CHECKPOINT_FOR_SUBMIT = epoch_005.pth
    


    === Val (image-level split, global gallery) ===
    mrr            : 0.8892
    ndcg           : 0.9116
    recall_at_1    : 0.8449
    recall_at_3    : 0.9254
    recall_at_5    : 0.9437
    recall_at_10   : 0.9629
    recall_at_50   : 0.9808
    l2_dist        : 1.0720


```python
# ============================================================
# 10) Submission (robusta e consistente con la validazione)
# ============================================================
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# -- 1) Carica il best checkpoint selezionato in valutazione
ckpt_dir = Path(MODEL_PATH).parent / "checkpoints"
def _pick_best_ckpt():
    # priorità: BEST_CHECKPOINT_FOR_SUBMIT se esiste; poi best per MRR se 'results' esiste; poi ultimo; infine MODEL_PATH
    if 'BEST_CHECKPOINT_FOR_SUBMIT' in globals() and BEST_CHECKPOINT_FOR_SUBMIT:
        return ckpt_dir / BEST_CHECKPOINT_FOR_SUBMIT
    if 'results' in globals() and isinstance(results, list) and len(results) > 0:
        name, _ = max(results, key=lambda x: x[1])
        return ckpt_dir / name
    ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
    if len(ckpts) > 0:
        return ckpts[-1]
    return Path(MODEL_PATH)

best_ckpt_path = _pick_best_ckpt()
print(f"[Submit] Carico checkpoint: {best_ckpt_path}")
state = torch.load(best_ckpt_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# -- 2) Carica test
test_data  = load_data("data/test/test.clean.npz")
test_ids   = test_data['captions/ids']
test_embds = torch.from_numpy(test_data['captions/embeddings']).float()  # (N, D_txt)

# -- 3) Predizione in batch (coerente con val)
@torch.no_grad()
def aggregate_slots(S_T: torch.Tensor, V_ref: torch.Tensor | None = None, mode: str = "mean"):
    assert S_T.dim() == 3
    if mode == "mean" or V_ref is None:
        return F.normalize(S_T.mean(dim=1), dim=-1)
    Vn = F.normalize(V_ref, dim=-1)
    sims = torch.einsum('bkd,bd->bk', S_T, Vn)
    idx  = sims.argmax(dim=1)
    out  = S_T[torch.arange(S_T.size(0), device=S_T.device), idx, :]
    return F.normalize(out, dim=-1)

pred_chunks = []
with torch.inference_mode():
    loader = DataLoader(test_embds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=='cuda'))
    for Xb in loader:
        Xb = Xb.to(DEVICE, non_blocking=True)
        S_Tb, _, _ = model(Xb)                     # (B, K, D)
        Eb = aggregate_slots(S_Tb, mode="mean")    # (B, D) L2-normalized
        pred_chunks.append(Eb.cpu().to(torch.float32))

pred_embds_test = torch.cat(pred_chunks, dim=0)   # (N, D) CPU float32

# -- 4) Sanity checks
assert pred_embds_test.ndim == 2, f"Got shape {tuple(pred_embds_test.shape)}"
assert len(test_ids) == pred_embds_test.size(0), f"ids({len(test_ids)}) != preds({pred_embds_test.size(0)})"

if not torch.isfinite(pred_embds_test).all():
    pred_embds_test = torch.nan_to_num(pred_embds_test, nan=0.0, posinf=1.0, neginf=-1.0)
    pred_embds_test = F.normalize(pred_embds_test, dim=-1)

# -- 5) (Opzionale) salva per analisi locale
np.save("pred_test_embeddings.npy", pred_embds_test.numpy())

# -- 6) Genera CSV
submission = generate_submission(test_ids, pred_embds_test, 'submission.csv')
print(f"[Submit] Usato checkpoint: {best_ckpt_path.name if best_ckpt_path.exists() else best_ckpt_path}")
print("Submission saved to: submission.csv")

```

    [Submit] Carico checkpoint: models\checkpoints\epoch_005.pth
    Generating submission file...
    ✓ Saved submission to submission.csv
    [Submit] Usato checkpoint: epoch_005.pth
    Submission saved to: submission.csv
    


```python
# ============================
# RIEPILOGO FINALE (SOLO STAMPA)
# ============================
import torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from challenge.src.eval.metrics import mrr, recall_at_k, ndcg

# ---- 1) Scegli il checkpoint da testare ----
# Se hai già 'results' (MRR per epoch) uso il migliore; altrimenti prendo l'ultimo disponibile.
try:
    best_ckpt_name, _ = max(results, key=lambda x: x[1])
    ckpt_path = (ckpt_dir / best_ckpt_name)
except Exception:
    ckpt_path = sorted((Path(MODEL_PATH).parent / "checkpoints").glob("epoch_*.pth"))[-1]

print("\n[Report] Uso checkpoint:", ckpt_path.name)

# ---- 2) Aggregatori (non toccano lo stato globale) ----
@torch.no_grad()
def agg_mean(S_T, t_vis_n=None):
    return F.normalize(S_T.mean(dim=1), dim=-1)

@torch.no_grad()
def agg_softmax_with_tvis(S_T, t_vis_n, beta: float):
    # conf_k = cos(S_k, t_vis)
    conf = torch.einsum('bkd,bd->bk', S_T, t_vis_n).clamp(-1, 1)  # (B,K)
    w = torch.softmax(beta * conf, dim=1).unsqueeze(-1)           # (B,K,1)
    z = (w * S_T).sum(dim=1)                                      # (B,D)
    return F.normalize(z, dim=-1)

@torch.no_grad()
def agg_topk_mean(S_T, t_vis_n, k: int):
    conf = torch.einsum('bkd,bd->bk', S_T, t_vis_n).clamp(-1, 1)  # (B,K)
    topk_idx = conf.topk(k, dim=1).indices                        # (B,k)
    B, D = S_T.size(0), S_T.size(-1)
    gather = S_T.gather(1, topk_idx.unsqueeze(-1).expand(B, k, D))# (B,k,D)
    return F.normalize(gather.mean(dim=1), dim=-1)

AGG_SPACE = [
    ("mean",            lambda S,T: agg_mean(S,T)),
    ("softmax_b6",      lambda S,T: agg_softmax_with_tvis(S,T,beta=6.0)),
    ("softmax_b8",      lambda S,T: agg_softmax_with_tvis(S,T,beta=8.0)),
    ("softmax_b10",     lambda S,T: agg_softmax_with_tvis(S,T,beta=10.0)),
    ("topk2_mean",      lambda S,T: agg_topk_mean(S,T,k=2)),
    ("topk3_mean",      lambda S,T: agg_topk_mean(S,T,k=3)),
]

# ---- 3) Istanzia un modello *temporaneo* per non alterare il tuo 'model' ----
tmp_model = SetPredictionHead(
    d_text=X.shape[1],
    d_vis=y.shape[1],
    K=K_SLOTS,
    hidden=2048,
    slot_dropout_p=SLOT_DROPOUT_P
).to(DEVICE)
tmp_state = torch.load(ckpt_path, map_location=DEVICE)
tmp_model.load_state_dict(tmp_state)
tmp_model.eval()

# ---- 4) Predizioni su validation per ogni aggregatore + metriche retrieval ----
def eval_agg(agg_name, agg_fn):
    preds = []
    with torch.no_grad():
        for Xb, _ in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False):
            Xb = Xb.to(DEVICE, non_blocking=True)
            S_Tb, _, tvis = tmp_model(Xb)
            zb = agg_fn(S_Tb, tvis)
            preds.append(zb.cpu())
    Z = F.normalize(torch.cat(preds, dim=0), dim=-1).cpu()        # (Nq,D)

    sims = Z @ val_img_embd.T
    topk_idx = torch.topk(sims, k=min(100, val_img_embd.size(0)), dim=1, largest=True, sorted=True).indices.numpy()
    res = {
        "MRR": mrr(topk_idx, val_label),
        "R@1": recall_at_k(topk_idx, val_label, 1),
        "R@5": recall_at_k(topk_idx, val_label, 5),
        "R@10": recall_at_k(topk_idx, val_label, 10),
        "NDCG": ndcg(topk_idx, val_label),
    }
    return res

print("\n[Val] Galleria immagini:", val_img_embd.size(0), " | Query (caption val):", len(val_dataset))

rows = []
for name, fn in AGG_SPACE:
    res = eval_agg(name, fn)
    rows.append((name, res["MRR"], res["R@1"], res["R@5"], res["R@10"], res["NDCG"]))

# Ordina per MRR desc e stampa tabella compatta
rows = sorted(rows, key=lambda r: r[1], reverse=True)
print("\n=== Aggregatori: classifica per MRR (validation) ===")
print(f"{'Agg':<14} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'NDCG':>7}")
for name, MRRv, R1, R5, R10, NDCGv in rows:
    print(f"{name:<14} {MRRv:7.4f} {R1:7.4f} {R5:7.4f} {R10:7.4f} {NDCGv:7.4f}")

best_name, best_mrr = rows[0][0], rows[0][1]
print(f"\n⇒ Miglior aggregatore (@{ckpt_path.name}): {best_name}  |  MRR={best_mrr:.5f}")

# ---- 5) Diagnostica geometrica (offdiag/log-var) sullo stesso checkpoint ----
@torch.no_grad()
def _diagnostics_batch():
    Xb, _ = next(iter(DataLoader(val_dataset, batch_size=min(256, len(val_dataset)), shuffle=True)))
    Xb = Xb.to(DEVICE)
    S_Tb, E_Tb, _ = tmp_model(Xb)
    E_Tn = F.normalize(E_Tb, dim=-1)
    K = E_Tn.size(1)
    C = torch.matmul(E_Tn, E_Tn.transpose(-1, -2))
    off = (C - torch.eye(K, device=E_Tn.device).unsqueeze(0)).abs().sum(dim=(1,2)) / (K*(K-1))
    var_fd = S_Tb.var(dim=1, unbiased=False).mean(dim=1)
    logvar = torch.log(var_fd.clamp_min(1e-8))
    return float(off.mean().item()), float(logvar.mean().item())

offdiag_mean, logvar_mean = _diagnostics_batch()
print(f"\n[Geometry] offdiag(E_T)≈{offdiag_mean:.4f} | log-var(S_T)≈{logvar_mean:.2f}")

# ---- 6) (Opzionale) Se è presente 'results', ristampa il best per-epoch già calcolato ----
if 'results' in globals() and isinstance(results, list) and len(results)>0:
    best_ckpt_prev, best_mrr_prev = max(results, key=lambda x: x[1])
    print(f"\n[Storico] Best per-epoch (MRR sui checkpoint): {best_ckpt_prev} → {best_mrr_prev:.5f}")

```

    
    [Report] Uso checkpoint: epoch_005.pth
    
    [Val] Galleria immagini: 2516  | Query (caption val): 12580
    
    === Aggregatori: classifica per MRR (validation) ===
    Agg                MRR     R@1     R@5    R@10    NDCG
    softmax_b10     0.4547  0.3188  0.6231  0.7358  0.5603
    softmax_b8      0.4545  0.3183  0.6236  0.7355  0.5601
    softmax_b6      0.4544  0.3180  0.6232  0.7359  0.5601
    mean            0.4541  0.3175  0.6229  0.7355  0.5598
    topk3_mean      0.4508  0.3157  0.6180  0.7288  0.5569
    topk2_mean      0.4475  0.3110  0.6180  0.7267  0.5538
    
    ⇒ Miglior aggregatore (@epoch_005.pth): softmax_b10  |  MRR=0.45470
    
    [Geometry] offdiag(E_T)≈0.1516 | log-var(S_T)≈-8.19
    
    [Storico] Best per-epoch (MRR sui checkpoint): epoch_005.pth → 0.45406
    


```python
# img mask
val_img_idx = np.where(IMG_VAL_MASK)[0]
train_img_idx = np.where(IMG_TRAIN_MASK)[0]
assert set(val_img_idx).isdisjoint(set(train_img_idx)), "Leak immagini!"

# caption→img check
cap_gt_img_idx = cap_gt_img_idx  # come da tuo codice
train_caps_img = set(cap_gt_img_idx[CAP_TRAIN_MASK])
val_caps_img   = set(cap_gt_img_idx[CAP_VAL_MASK])
overlap_imgs = train_caps_img & val_caps_img
print("Overlap immagini train/val (dovrebbe essere 0):", len(overlap_imgs))

```

    Overlap immagini train/val (dovrebbe essere 0): 0
    
