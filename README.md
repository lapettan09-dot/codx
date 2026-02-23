# Sunbath Style Dress Transformer

This project is a small AI app for fashion-contest prototyping. It transforms an uploaded clothing photo into an **American-style sunbathing suitable dress concept** using an image-to-image diffusion pipeline.

## Features

- Upload any subject/dress image
- Add optional style guidance
- Tune generation with strength, guidance scale, inference steps, and seed
- Generate transformed outputs using a locally run diffusion model
- Includes a built-in, fashion-focused default prompt

## Stack

- Python
- Gradio UI
- Hugging Face Diffusers
- Stable Diffusion image-to-image pipeline

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the local URL printed by Gradio.

## Notes

- First run downloads the model weights and can take time.
- For best quality and speed, use a GPU-enabled environment.
- The default model is `runwayml/stable-diffusion-v1-5` and can be overridden via environment variable:

```bash
export MODEL_ID="runwayml/stable-diffusion-v1-5"
```

## Responsible Use

This tool is intended for creative fashion visualization and contest ideation only.
Always ensure you have permission to process images, avoid generating harmful or deceptive content, and comply with applicable rules and laws.
