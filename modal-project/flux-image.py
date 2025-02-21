import io
import modal
import random
import numpy as np

MODEL_PATH = "/models"
MINUTES = 60
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

app = modal.App()

container_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "fastapi",
        "accelerate",
        "hf_transfer",
        "git+https://github.com/huggingface/diffusers.git",
        "invisible_watermark",
        "torch",
        "transformers==4.42.4",
        "xformers",
        "sentencepiece",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_PATH,
        }
    )
)

with container_image.imports():
    from diffusers import FluxPipeline
    import torch
    from fastapi import Response


@app.cls(
    image=container_image,
    gpu="H100",
    timeout=10 * MINUTES,
    volumes={MODEL_PATH: cache_volume},
)
class Inference:

    @modal.enter()
    def load_pipeline(self):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")  # type: ignore

    @modal.method()
    def run(
        self,
        prompt: str,
        seed=42,
        randomize_seed=False,
        width=1024,
        height=1024,
        num_inference_steps=4,
    ):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator().manual_seed(seed)
        images = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=0.0,
            max_sequence_length=256,
        ).images
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        torch.cuda.empty_cache()  # reduce fragmentation
        return image_output

    @modal.web_endpoint(docs=True)
    def web(self, prompt: str, seed: int = None):
        return Response(
            content=self.run.local(  # run in the same container
                prompt, batch_size=1, seed=seed
            )[0],
            media_type="image/png",
        )
