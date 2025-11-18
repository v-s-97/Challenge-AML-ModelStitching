# STM — Cross-Modal Retrieval (AML 2025/26)

This project implements a text-to-image retrieval system extending VSE++ with a compositional *slot* branch to improve fine-grained alignment between captions and visual features.

## Method

* **TextToVis Adapter**: two-layer MLP mapping RoBERTa text embeddings (1024-D) to the visual space (1536-D, DINO-VAE).
* **SlotAuxHead**: generates four normalized slot vectors capturing local semantic components.
* Only the global embedding **Z** is used at test time; slots are auxiliary during training.
* Final system uses an **ensemble** of identical models with different seeds.

## Losses

Combination of:

* Global triplet loss
* Max-over-slot triplet loss
* Per-slot InfoNCE
* Intra-slot diversity regularization
* Condensation loss aligning Z with the most relevant slot

## Training

* Adam, 24 epochs, batch size 128, margin 0.20
* Loss weights: 0.30 (global triplet), 0.15 (slot triplet), 0.02 (ISDL), 0.05 (condensation)

## Results

* **MRR = 0.88141** on the public leaderboard.
* The model uses slot-based supervision during training while keeping a single compact embedding for deployment.

## Repository Structure

* `STM.ipynb` — main notebook

## Authors
    Luca Moresca
    Valerio Santini
    Nicholas Suozzi