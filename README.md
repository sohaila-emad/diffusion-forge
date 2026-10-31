# DiffusionForge
### Stable Diffusion 1.5 from Scratch | Text-to-Image Generation & Prompt-Guided Image Editing

**DiffusionForge** is a hands-on implementation of Stable Diffusion 1.5 built from the ground up using **Python and PyTorch**. The project aims to go beyond simply using pretrained generative AI models by reconstructing the core architecture and computational pipeline that power latent diffusion models.

From text encoding and latent-space representation to noise prediction and image reconstruction, each major component is implemented and integrated into a unified image-generation pipeline.

Rather than relying on high-level diffusion libraries to perform the generation process, this project explores the underlying mechanisms of Stable Diffusion through custom-built modules, using pretrained Stable Diffusion 1.5 weights to bring the implementation to life.

### ✨ Key Features

- **Text Encoding:** CLIP tokenizer and text encoder implementation to transform natural-language prompts into contextual embeddings.
- **Variational Autoencoder (VAE):** Custom encoder and decoder implementations for mapping images into and reconstructing them from a compressed latent space.
- **U-Net Architecture:** Implementation of the diffusion U-Net, including residual blocks, self-attention, cross-attention, and time embeddings.
- **Attention Mechanisms:** Custom multi-head self-attention and cross-attention modules to capture spatial relationships and align image representations with text conditioning.
- **Diffusion Process:** Implementation of the denoising process and sampling pipeline to progressively transform random noise into meaningful latent representations.
- **Classifier-Free Guidance:** Conditional and unconditional predictions combined to improve alignment between generated images and text prompts.
- **Image-to-Image Generation:** Prompt-guided image transformation by encoding an input image into latent space and controlling the diffusion process through the noise strength.
- **Custom Generation Pipeline:** Integration of all components into an end-to-end pipeline for text-to-image generation and image-to-image transformation.

### 🧠 Technical Stack

- Python
- PyTorch
- NumPy
- Pillow
- Hugging Face model checkpoints

### 🎯 Project Objective

The main objective is to understand and implement the internal workings of latent diffusion models at an architectural and mathematical level, gaining practical experience with deep learning, generative modeling, attention mechanisms, latent representations, and PyTorch model development.

**Built to understand diffusion models from the inside out, not just generate images with them.**
