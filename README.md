# 🔥 DiffusionForge
### Stable Diffusion 1.5 From Scratch | Text-to-Image Generation & Image-to-Image Transformation

**Building diffusion models from the inside out.**

DiffusionForge is a hands-on implementation of **Stable Diffusion 1.5 using Python and PyTorch**, developed to explore and reconstruct the core architectures and mathematical mechanisms behind latent diffusion models.

Rather than relying on high-level diffusion frameworks or pre-built generation pipelines, this project implements the essential components of Stable Diffusion individually and integrates them into a custom end-to-end inference pipeline.

From CLIP text encoding and latent-space representations to U-Net noise prediction, attention mechanisms, and iterative denoising, the project explores how natural-language prompts are transformed into detailed visual outputs.

The implementation uses pretrained Stable Diffusion 1.5 checkpoint weights while reconstructing the model architecture and inference workflow independently.

---

## ✨ Features

- **Text-to-Image Generation:** Generate images from natural-language prompts using a custom Stable Diffusion pipeline.
- **Image-to-Image Transformation:** Transform existing images according to text prompts while controlling how much of the original image is preserved.
- **CLIP Text Encoder:** Custom implementation of the CLIP text encoder, including token embeddings, positional embeddings, and transformer layers.
- **Variational Autoencoder (VAE):** Custom encoder and decoder for converting images into compressed latent representations and reconstructing them.
- **U-Net Architecture:** Implementation of the diffusion U-Net with residual blocks, attention mechanisms, and time conditioning.
- **Attention Mechanisms:** Multi-head self-attention and cross-attention to model spatial relationships and connect textual conditioning with image representations.
- **Classifier-Free Guidance (CFG):** Combine conditional and unconditional noise predictions to improve prompt alignment.
- **DDPM Sampling:** Implement the iterative denoising process used to generate latent representations from noise.
- **Custom Inference Pipeline:** Integrate all components into a unified generation workflow without relying on a ready-made Stable Diffusion pipeline.

---

## 🧠 Architecture & Implementation

The project reconstructs the major building blocks of Stable Diffusion 1.5:

| Component | Implementation |
|---|---|
| Tokenizer | CLIP Tokenizer |
| Text Encoder | Custom CLIP Transformer |
| Image Encoder | Custom VAE Encoder |
| Image Decoder | Custom VAE Decoder |
| Denoising Network | Custom U-Net |
| Attention | Self-Attention & Cross-Attention |
| Sampling | DDPM |
| Guidance | Classifier-Free Guidance |
| Pipeline | Custom end-to-end inference pipeline |

### 1. Text Encoding

Natural-language prompts are tokenized into sequences of 77 tokens and passed through the CLIP text encoder to produce contextual embeddings.

### 2. Latent Representation

The VAE encoder maps input images into a compressed latent space, allowing the diffusion process to operate on latent representations rather than full-resolution images.

### 3. Noise Prediction

The U-Net predicts the noise present in latent representations at different diffusion timesteps.

Its architecture includes:
- Residual blocks
- Time embeddings
- Multi-head self-attention
- Text-conditioned cross-attention
- GEGLU feed-forward layers

### 4. Iterative Denoising

The sampling process progressively removes noise from the latent representation, guided by the text conditioning and classifier-free guidance.

### 5. Image Reconstruction

The VAE decoder transforms the final denoised latent representation back into the image space.

---

## 🎨 Generation Results

### Text-to-Image Generation

**Prompt:**

> A white cat with wide eyes standing on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.

**Configuration:**

| Parameter | Value |
|---|---|
| Model | Stable Diffusion 1.5 |
| Sampler | DDPM |
| Inference Steps | 50 |
| CFG Scale | 8 |
| Seed | 42 |
| Classifier-Free Guidance | Enabled |

**Generated Output:**

<img width="512" height="512" alt="generated image" src="https://github.com/user-attachments/assets/d1973073-a229-4768-a0b6-1d5849371aad" />


---

### Image-to-Image Transformation

The image-to-image pipeline takes an existing image and a text prompt, then generates a transformed result through controlled diffusion in latent space.

**Prompt:**

> Make the flower blue and the grass pink, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.

**Configuration:**

| Parameter | Value |
|---|---|
| Input | User-provided image |
| Sampler | DDPM |
| Inference Steps | 50 |
| CFG Scale | 8 |
| Seed | 42 |
| Strength | 0.9 |

**Input Image:**

<img width="267" height="216" alt="WhatsApp Image 2026-10-19 at 1 39 32 AM" src="https://github.com/user-attachments/assets/e0d2b42c-bf10-4d52-a294-fd193f119078" />


**Generated Output:**

<img width="512" height="512" alt="image to image" src="https://github.com/user-attachments/assets/a777671e-5212-4170-8801-295e9a5430e4" />


---

## 🛠️ Technology Stack

- **Python** – Core implementation
- **PyTorch** – Neural network architectures and tensor computations
- **Hugging Face Transformers** – CLIP tokenizer
- **Hugging Face Hub** – Pretrained checkpoint retrieval
- **NumPy** – Numerical operations
- **Pillow (PIL)** – Image loading and processing
- **Google Colab** – Cloud-based GPU execution

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/diffusion-forge.git
cd diffusion-forge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the pretrained checkpoint

The implementation uses the Stable Diffusion 1.5 checkpoint:

`stable-diffusion-v1-5/stable-diffusion-v1-5`

The checkpoint can be downloaded directly using Hugging Face Hub:

```python
from huggingface_hub import hf_hub_download

model_file = hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.ckpt"
)
```

### 4. Initialize the tokenizer

```python
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer(
    vocab="data/vocab.json",
    merges="data/merges.txt"
)
```

### 5. Load pretrained weights

```python
models = model_loader.preload_models_from_standard_weights(
    model_file,
    "cpu"
)
```

### 6. Run the generation pipeline

Configure the prompt, sampling parameters, input image (if applicable), and execute the custom `pipeline.generate()` function.

**Note:** A CUDA-enabled GPU is recommended for practical inference. Google Colab can be used as a cloud GPU environment.

---

## 📚 What This Project Explores

This implementation provided practical experience with:

- Generative AI and latent diffusion models
- Transformer architectures and contextual embeddings
- Variational Autoencoders
- U-Net architecture and residual learning
- Multi-head attention and cross-attention
- Noise schedules and iterative denoising
- Classifier-Free Guidance
- PyTorch module design and pretrained state-dict integration
- GPU-based deep learning inference

---

## 🎯 Project Goal

The goal of DiffusionForge is to understand the engineering and mathematical foundations of latent diffusion models by implementing their core components, integrating pretrained weights, and building a functional image-generation system.

**Not just using Stable Diffusion. Understanding how it works, one component at a time.**

---
