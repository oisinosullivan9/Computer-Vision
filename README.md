# Diffusion-Based Saliency Prediction

- **Course**: CS6524 Computer Vision Project
- **Objective**: Generate saliency maps from RGB inputs using a diffusion model trained on SALICON.

## Project Overview
- Conditional diffusion (DDPM) pipeline that learns to predict human visual attention maps from SALICON images.
- Training, validation, and inference logic live in `notebooks/salicon_diffusion.ipynb`.
- Outputs include per-epoch checkpoints, validation visualizations, and test-set saliency predictions.

## Environment & Dataset Setup
- **Prerequisites**: Python 3.9+, Git LFS, CUDA-capable GPU recommended (but CPU works for inspection).
- **Clone with data**:
	1. `git clone <repo-url>`
	2. `cd Computer-Vision`
	3. `git lfs pull` (downloads the SALICON splits committed via LFS into `data/`).
- **Virtual environment** (example using `venv`):
	1. `python -m venv .cv_venv`
	2. `source .cv_venv/bin/activate`
	3. `pip install -r requirements.txt`
- **Dependencies**: install the notebook requirements – `pip install torch torchvision numpy pillow matplotlib tqdm`
	- If you plan to train on GPU, match the PyTorch build to your CUDA version (`pip install torch --index-url https://download.pytorch.org/whl/cu118` etc.).
	- Optional: `pip install jupyter ipykernel` then register the kernel (`python -m ipykernel install --user --name saliency-diffusion`).

## Repository Layout
- `data/`: SALICON images (`train/`, `val/`, `test/`) and fixation JSONs (`fixations_train2014.json`, `fixations_val2014.json`).
- `notebooks/salicon_diffusion.ipynb`: main training and inference notebook.
- `notebooks/saliency_diffusion_outputs/`: auto-created directory tree for checkpoints and visual artifacts.

## Running the Notebook
- Launch Jupyter (`jupyter lab` or `code notebooks/salicon_diffusion.ipynb`).
- Execute cells sequentially:
	- Cells 1–6 prepare configuration, generate ground truth saliency maps, and visualize cached samples.
	- Cells 7–11 define the model (UNet + DDPM) and training utilities.
	- Cell 12 runs the training loop, saving `last.pt`, `saliency_diffusion_unet.pt`, and per-epoch checkpoints under `notebooks/saliency_diffusion_outputs/`.
	- Cell 13 optionally restores optimizer/scheduler states if resuming.
	- Cell 14 loads the best checkpoint into the diffusion wrapper.
	- Cell 15 samples validation batches for qualitative inspection.
	- Cells 16–17 build a test DataLoader and run inference on 10 random test images (saving predictions).
- Adjust `Config` (cell 1) to change paths, image sizes, batch size, or diffusion hyperparameters before re-running.

## Notebook Outputs
- `notebooks/saliency_diffusion_outputs/last.pt`: latest checkpoint payload (model + optimizer + scheduler).
- `notebooks/saliency_diffusion_outputs/saliency_diffusion_unet.pt`: best-validation checkpoint.
- `notebooks/saliency_diffusion_outputs/epochs/epoch_XXX.pt`: per-epoch snapshots.
- `notebooks/saliency_diffusion_outputs/samples/`: validation visualizations (image, GT saliency, prediction).
- `notebooks/saliency_diffusion_outputs/test_predictions/`: `.npy` saliency arrays and `.png` side-by-side comparisons for sampled test images.

## Model Architecture (TBD)
- **Backbone**: UNet with residual blocks and GroupNorm, conditioned on RGB inputs concatenated with noisy saliency latents.
- **Positional Encoding**: sinusoidal time embeddings projected through an MLP to modulate residual blocks.
- **Diffusion Process**: deterministic forward diffusion (linear beta schedule) and learned reverse process via DDPM sampling.
- **Loss Mix**: noise MSE, L1, MSE, and lightweight SSIM on the predicted clean saliency to encourage sharp yet stable reconstructions.

## Results (TBD)
- Metrics: to be populated with comparisons between predicted saliency maps and ground-truth fixation-derived maps (e.g., Pearson correlation, NSS, KL).
- Visuals: qualitative grids for validation and test splits to be inserted after final evaluation.
