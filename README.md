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
	1. `python -m venv .venv`
	2. `source .venv/bin/activate`
	3. `pip install -r requirements.txt`

## Repository Layout
- `data/`: SALICON images (`train/`, `val/`, `test/`) and fixation JSONs (`fixations_train2014.json`, `fixations_val2014.json`).
- `notebooks/salicon_diffusion.ipynb`: main training and inference notebook.
- `notebooks/saliency_diffusion_outputs/`: auto-created directory tree for checkpoints and visual artifacts.
- `weights/`: best performing weights generated from experiemnts

## Running the Notebook
- Launch Jupyter.
- Execute cells sequentially
- Adjust `Config` (cell 1) to change paths, image sizes, batch size, or diffusion hyperparameters before re-running.

## Notebook Outputs
- `notebooks/saliency_diffusion_outputs/last.pt`: latest checkpoint payload (model + optimizer + scheduler).
- `notebooks/saliency_diffusion_outputs/saliency_diffusion_unet.pt`: best-validation checkpoint.
- `notebooks/saliency_diffusion_outputs/epochs/epoch_XXX.pt`: per-epoch snapshots.
- `notebooks/saliency_diffusion_outputs/samples/`: validation visualizations (image, GT saliency, prediction).
- `notebooks/saliency_diffusion_outputs/test_predictions/`: `.npy` saliency arrays and `.png` side-by-side comparisons for sampled test images.

## Model Architecture
- **Backbone**: UNet with residual blocks and GroupNorm, conditioned on RGB inputs concatenated with noisy saliency latents.
- **Positional Encoding**: sinusoidal time embeddings projected through an MLP to modulate residual blocks.
- **Diffusion Process**: deterministic forward diffusion (linear beta schedule) and learned reverse process via DDPM sampling.
- **Loss Mix**: noise MSE, L1, MSE, and lightweight SSIM on the predicted clean saliency to encourage sharp yet stable reconstructions.

## Results

| Metric     | Mean   | Std    |
|------------|--------|--------|
| CC         | 0.7185 | 0.1363 |
| NSS        | 2.6240 | 1.4362 |
| AUC-Judd   | 0.9276 | 0.0606 |
| AUC-Borji  | 0.9002 | 0.0905 |

- **CC** (Correlation Coefficient): Pearson correlation with ground truth saliency map
- **NSS** (Normalized Scanpath Saliency): Mean normalized saliency at fixation points
- **AUC-Judd**: Area under ROC curve using all non-fixated pixels as negatives
- **AUC-Borji**: AUC with randomly sampled non-fixated negatives (shuffled)