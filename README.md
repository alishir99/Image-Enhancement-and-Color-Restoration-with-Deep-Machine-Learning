# Deep-Image-Denoising-and-Colorization
## Jigsaw Puzzle Solver & Image Colorization

This repository contains code and models for solving jigsaw puzzles, denoising reconstructed images, and colorizing grayscale images using deep learning. The project is organized for easy experimentation and reproducibility.

## Features

- **Jigsaw Puzzle Solver:** Deep learning models to reconstruct images from shuffled pieces.
- **Denoising:** UNet-based denoiser to clean up noisy grayscale reconstructions.
- **Colorization:** Automatic colorization of grayscale images using a UNet model.
- **Batch Processing:** Scripts to process entire folders of images for denoising and colorization.

## Project Structure

```
Jigsaw_project_final/
│
├── color.py
├── colorization.ipynb
├── deniose_test.ipynb
├── denoiser.py
├── jigsaw_model_gray_noise.pth
├── jigsaw-puzzle-solver.ipynb
├── dataset/
│   ├── test/
│   └── train/
│
├── jigsaw_result/
│   └── reconstructed_*.png
│
├── lightning_logs/
├── models/
├── results/
```

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib
- deeplay (for UNet2d, if using denoiser)

Install dependencies:
```sh
pip install torch torchvision pillow matplotlib
# If using deeplay:
pip install deeplay
```

### Usage

#### 1. Jigsaw Puzzle Solving

See [`jigsaw-puzzle-solver.ipynb`](jigsaw-puzzle-solver.ipynb) for training and evaluating the jigsaw puzzle model.

#### 2. Denoising

Use [`deniose_test.ipynb`](deniose_test.ipynb) to denoise reconstructed images:

- Loads a trained UNet denoiser.
- Processes images in `results/jigsaw/reconstructed/`.
- Saves denoised images to `results/denoiser/`.

#### 3. Colorization

Use [`colorization.ipynb`](colorization.ipynb) to colorize grayscale images:

- Loads a trained UNet colorizer.
- Processes single images or batches.
- Example usage:
    ```python
    colorize_and_save('results/denoiser/Image_1313_reconstructed_denoised.png', 'output_colorized.png')
    ```
#### 4. Models

- Pretrained models are stored as `.pth` files (e.g., `jigsaw_model_gray_noise.pth`).
- Place your trained models in the `models/` directory for easy loading.
#### 5. Dataset 
- The dataset used is called "Butter Flies Images Dataset" on kaggle.
- https://www.kaggle.com/datasets/meetnagadia/butterflies-dataset
## Results

- Denoised and colorized images are saved in the `results/` and `output_colorized.png`.
- Example outputs can be found in `jigsaw_result/` and `results/`.

## Acknowledgements

- UNet and denoising code adapted from [deeplay](https://github.com/DeepTrackAI/deeplay).
- Puzzle and colorization methods inspired by recent deep learning research.

---

**Author:** [Ali Shirzad]  

