import os
from functools import lru_cache

import gradio as gr
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

DEFAULT_PROMPT = (
    "Transform the outfit into an American-style sunbath suitable dress, "
    "elegant beachwear aesthetic, stylish but practical, flattering silhouette, "
    "summer palette, high-detail fashion photography"
)

NEGATIVE_PROMPT = (
    "nsfw, nudity, bad anatomy, blurry, low quality, distorted face, "
    "extra limbs, watermark, text"
)


@lru_cache(maxsize=1)
def load_pipeline(model_id: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe


def transform_dress(
    image: Image.Image,
    custom_style: str,
    strength: float,
    guidance_scale: float,
    steps: int,
):
    if image is None:
        raise gr.Error("Please upload an image.")

    model_id = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5")
    pipe = load_pipeline(model_id)

    prompt = DEFAULT_PROMPT
    if custom_style.strip():
        prompt = f"{prompt}, additional style: {custom_style.strip()}"

    image = image.convert("RGB").resize((768, 768))

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    )

    return result.images[0]


def build_ui():
    with gr.Blocks(title="Sunbath Dress Transformer") as demo:
        gr.Markdown(
            "## AI Fashion App: American-Style Sunbath Dress Transformer\n"
            "Upload an outfit photo and generate a sunbath-suitable American-style dress concept."
        )

        with gr.Row():
            input_image = gr.Image(type="pil", label="Input dress / subject image")
            output_image = gr.Image(type="pil", label="Transformed output")

        custom_style = gr.Textbox(
            label="Optional style directions",
            placeholder="e.g., resort luxury, retro Miami, vibrant floral print",
        )

        with gr.Row():
            strength = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Transformation strength")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")

        generate_btn = gr.Button("Generate Sunbath Dress", variant="primary")

        generate_btn.click(
            fn=transform_dress,
            inputs=[input_image, custom_style, strength, guidance, steps],
            outputs=[output_image],
        )

        gr.Markdown(
            "### Safety note\n"
            "Use only with images you are authorized to process. Keep outputs respectful and contest-compliant."
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
