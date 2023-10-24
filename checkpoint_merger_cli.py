"""
Stable Diffusion Checkpoint Merger CLI 
==================================
This module provides functionalities to merge Stable Diffusion models and perform inferences.

Credits:
-------
- Modified from: https://github.com/painebenjamin/app.enfugue.ai/blob/main/src/python/enfugue/diffusion/util/model_util.py
- Inspired by: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/dev/modules/extras.py
"""


import os
import re
import gc
import argparse
import random
import torch
import safetensors.torch
from PIL import Image
from typing import Optional, Union, Literal, Dict, cast

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
class ModelMerger:
    """
    Allows merging various Stable Diffusion models of various sizes.
    Inspired by https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/extras.py
    """

    CHECKPOINT_DICT_REPLACEMENTS = {
        "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
        "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
        "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
    }

    CHECKPOINT_DICT_SKIP = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

    discard_weights: Optional[re.Pattern]

    def __init__(
            self,
            primary_model: str,
            secondary_model: Optional[str],
            tertiary_model: Optional[str],
            interpolation: Optional[Literal["add-difference", "weighted-sum"]] = None,
            multiplier: Union[int, float] = 1.0,
            half: bool = True,
            discard_weights: Optional[Union[str, re.Pattern]] = None,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.tertiary_model = tertiary_model
        self.interpolation = interpolation
        self.multiplier = multiplier
        self.half = half

        self.discard_weights = re.compile(discard_weights) if isinstance(discard_weights, str) else discard_weights

    @staticmethod
    def as_half(tensor: torch.Tensor) -> torch.Tensor:
        """Halves a tensor if necessary"""
        return tensor.half() if tensor.dtype == torch.float else tensor

    @staticmethod
    def get_difference(theta0: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
        """Returns the difference between two tensors."""
        return theta0 - theta1

    @staticmethod
    def weighted_sum(theta0: torch.Tensor, theta1: torch.Tensor, alpha: Union[int, float]) -> torch.Tensor:
        """Returns the weighted sum of two tensors."""
        return ((1 - alpha) * theta0) + (alpha * theta1)

    @staticmethod
    def add_weighted_difference(theta0: torch.Tensor, theta1: torch.Tensor, alpha: Union[int, float]) -> torch.Tensor:
        """Adds a weighted difference back to the original tensor."""
        return theta0 + (alpha * theta1)

    @staticmethod
    def get_state_dict_from_checkpoint(checkpoint: Dict) -> Dict:
        """Extracts the state dictionary from the checkpoint."""
        state_dict = checkpoint.pop("state_dict", checkpoint)
        state_dict.pop("state_dict", None)  # Remove any sub-embedded state dicts

        transformed_dict = {
            ModelMerger.transform_checkpoint_key(key): value for key, value in state_dict.items()
        }
        return transformed_dict

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """Loads the checkpoint's state dictionary."""
        _, ext = os.path.splitext(checkpoint_path)
        print(f"Loading {checkpoint_path}...")
        checkpoint = safetensors.torch.load_file(
            checkpoint_path, device="cpu") if ext.lower() == ".safetensors" else torch.load(checkpoint_path, map_location="cpu")
        return ModelMerger.get_state_dict_from_checkpoint(checkpoint)

    @staticmethod
    def is_ignored_key(key: str) -> bool:
        """Checks if a key should be ignored during merge."""
        return "model" not in key or key in ModelMerger.CHECKPOINT_DICT_SKIP

    @staticmethod
    def transform_checkpoint_key(text: str) -> str:
        """Transforms a checkpoint key if needed."""
        for key, value in ModelMerger.CHECKPOINT_DICT_REPLACEMENTS.items():
            if key.startswith(text):
                return value + text[len(key):]
        return text

    def merge_models(self, output_path: str) -> None:
        """Runs the configured merger."""

        secondary_state_dict = None if not self.secondary_model else self.load_checkpoint(self.secondary_model)
        tertiary_state_dict = None if not self.tertiary_model else self.load_checkpoint(self.tertiary_model)

        theta_1 = secondary_state_dict

        if self.interpolation == "add-difference":
            if theta_1 is None or tertiary_state_dict is None:
                raise ValueError(f"{self.interpolation} requires three models.")
            print("Merging secondary and tertiary models.")
            for key in theta_1.keys():
                if self.is_ignored_key(key):
                    continue
                if key in tertiary_state_dict:
                    theta_1[key] = self.get_difference(theta_1[key], tertiary_state_dict[key])
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])
            del tertiary_state_dict
            gc.collect()

        interpolate = self.add_weighted_difference if self.interpolation == "add-difference" else self.weighted_sum

        theta_0 = self.load_checkpoint(self.primary_model)

        if theta_1:
            print("Merging primary and secondary models.")
            for key in theta_0.keys():
                if key not in theta_1 or self.is_ignored_key(key):
                    continue

                a, b = theta_0[key], theta_1[key]

                # Check for different model types based on channel count
                if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                    self._handle_different_model_types(a, b, key, theta_0, interpolate)
                else:
                    theta_0[key] = interpolate(a, b, self.multiplier)

                if self.half:
                    theta_0[key] = self.as_half(theta_0[key])

            del theta_1

        if self.discard_weights:
            theta_0 = {key: value for key, value in theta_0.items() if not re.search(self.discard_weights, key)}

        _, extension = os.path.splitext(output_path)
        if extension.lower() == ".safetensors":
            safetensors.torch.save_file(theta_0, output_path)
        else:
            torch.save(theta_0, output_path)
            
        print(f"Saving Merged Model:")
        print(f"  - Path: {output_path}")
        print(f"  - File Format: {'SafeTensors' if output_path.endswith('.safetensors') else 'PyTorch'}\n")

    def _handle_different_model_types(self, a, b, key, theta_0, interpolate):
        """Handles the case when merging different types of models."""
        if a.shape[1] == 4 and (b.shape[1] == 9 or b.shape[1] == 8):
            raise RuntimeError(
                "When merging different types of models, the primary model must be the specialized one."
            )

        if a.shape[1] == 8 and b.shape[1] == 4:
            theta_0[key][:, 0:4, :, :] = interpolate(a[:, 0:4, :, :], b, self.multiplier)
        else:
            assert a.shape[1] == 9 and b.shape[1] == 4, f"Unexpected dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
            theta_0[key][:, 0:4, :, :] = interpolate(a[:, 0:4, :, :], b, self.multiplier)

class Inference:
    def __init__(self, model_weights, sdxl=False):
        if sdxl:
            self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_weights,
                vae=self.vae,
                use_safetensors=True,
            ).to('cuda')
        else:
            self.vae = AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")
            self.pipe = StableDiffusionPipeline.from_single_file(
                model_weights,
                vae=self.vae,
                use_safetensors=True,
            ).to('cuda')
        if not disable_torch_compile:
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)


    @staticmethod
    def process_prompt_args(prompt: str, sdxl=False):
        prompt_args = prompt.split(" --")
        prompt = prompt_args[0]
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        num_inference_steps = 30
        width = height = 1024
        guidance_scale = 12
        seed = None
        images_per_prompt = 1
        
        for parg in prompt_args:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                width = int(m.group(1))
                continue
            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                height = int(m.group(1))
                continue
            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                seed = int(m.group(1))
                continue
            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                num_inference_steps = max(1, min(1000, int(m.group(1))))
                continue
            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                guidance_scale = float(m.group(1))
                continue
            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                negative_prompt = m.group(1)
                continue
            m = re.match(r"t (\d+)", parg, re.IGNORECASE)
            if m:
                images_per_prompt = int(m.group(1))
                continue
                
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        height = max(64, height - height % 8)
        width = max(64, width - width % 8)
        
        return prompt, negative_prompt, num_inference_steps, width, height, guidance_scale, seed, images_per_prompt
        
    def get_scheduler(self, name="Euler a"):
        # Get scheduler
        match name:
            case "DPM++ 2M":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config
                )

            case "DPM++ 2M Karras":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DPM++ 2M SDE":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++"
                )

            case "DPM++ 2M SDE Karras":
                return DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++",
                )

            case "DPM++ SDE":
                return DPMSolverSinglestepScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "DPM++ SDE Karras":
                return DPMSolverSinglestepScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DPM2":
                return KDPM2DiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "DPM2 Karras":
                return KDPM2DiscreteScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "Euler":
                return EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "Euler a":
                return EulerAncestralDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "Heun":
                return HeunDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "LMS":
                return LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                )

            case "LMS Karras":
                return LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config, use_karras_sigmas=True
                )

            case "DDIM":
                return DDIMScheduler.from_config(self.pipe.scheduler.config)

            case "DEISMultistep":
                return DEISMultistepScheduler.from_config(self.pipe.scheduler.config)

            case "UniPCMultistep":
                return UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
                
        return EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)  # default to "Euler a"

    def validate(self, prompt_string, image_output, sdxl=False):
        self.pipe.scheduler = self.get_scheduler()
        
        prompt, negative_prompt, num_inference_steps, width, height, guidance_scale, seed, num_inference_steps, num_images_per_prompt = self.process_prompt_args(prompt_string, sdxl=sdxl)

        generator = torch.Generator().manual_seed(seed)
        
        print("\nInference Parameters:")
        print(f"  - Prompt: {prompt}")
        print(f"  - Negative Prompt: {negative_prompt}")
        print(f"  - Image Width: {width}")
        print(f"  - Image Height: {height}")
        print(f"  - Num Inference Steps: {num_inference_steps}")
        print(f"  - Guidance Scale: {guidance_scale}")
        print(f"  - Images per Prompt: {num_images_per_prompt}\n")

        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        ).images

        return image
        
def parse_arguments():
    """Parses command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Merge and optionally validate Stable Diffusion models.")
    
    parser.add_argument("primary_model", help="Path to the primary model checkpoint.")
    parser.add_argument("output_path", help="Path to save the merged model.")
    parser.add_argument("--secondary_model", help="Path to the secondary model checkpoint.")
    parser.add_argument("--tertiary_model", help="Path to the tertiary model checkpoint.")
    parser.add_argument("--interpolation", choices=["add-difference", "weighted-sum"], help="Interpolation method to use.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Multiplier for the interpolation.")
    parser.add_argument("--half", action="store_true", help="Halve the tensor if necessary.")
    parser.add_argument("--discard_weights", help="Pattern of weights to discard.")
    parser.add_argument("--prompt", help="Positive prompts for the validation.")
    parser.add_argument("--image_output", help="Path to save the generated image.")
    parser.add_argument("--sampler", default="Euler a", help="Scheduler sampler for the validation. Defaults to 'Euler a'.")
    parser.add_argument("--sdxl", action="store_true", help="Use StableDiffusionXLPipeline instead of StableDiffusionPipeline.")
    parser.add_argument("--disable_torch_compile", action="store_true", help="Disable torch.compile for the unet model.")
    
    return parser.parse_args()

def checkpoint_merger(
    primary_model: str,
    output_path: str,
    secondary_model: Optional[str] = None,
    tertiary_model: Optional[str] = None,
    interpolation: Optional[Literal["add-difference", "weighted-sum"]] = None,
    multiplier: float = 1.0,
    half: bool = False,
    discard_weights: Optional[str] = None,
    validate: bool = False,
    prompt: Optional[str] = None,
    image_output: Optional[str] = None,
    sampler: str = "Euler a",
    sdxl: bool = False,
    disable_torch_compile: bool = False,
):
    # Notify users about loading the checkpoint
    print("Loading checkpoint...")
    print(f"Checkpoint: {primary_model}\n")
    
    merger = ModelMerger(
        primary_model=primary_model,
        secondary_model=secondary_model,
        tertiary_model=tertiary_model,
        interpolation=interpolation,
        multiplier=multiplier,
        half=half,
        discard_weights=discard_weights
    )

    print("Merging Models:")
    print(f"  - Primary Model:   {primary_model}")
    print(f"  - Secondary Model: {secondary_model}")
    print(f"  - Tertiary Model:  {tertiary_model}")
    print(f"Interpolation Method: {interpolation}")
    print(f"Multiplier: {multiplier}")
    print(f"Tensor Halving: {'Enabled' if half else 'Disabled'}")
    print(f"Discard Weights: {discard_weights}\n")

    merger.merge_models(output_path)
    free_memory()
    
    if prompt:
        infer = Inference(output_path, sdxl=sdxl)
        image = infer.validate(prompt, image_output, sdxl=sdxl)
        free_memory()
    
        if image_output:
            basename = os.path.splitext(os.path.basename(image_output))[0]
            dirpath = os.path.dirname(image_output)
            
            if isinstance(images, list):
                for idx, img in enumerate(images):
                    img.save(os.path.join(dirpath, f'{basename}_{idx}.png'))
            else:
                images.save(image_output)
    
        return images

def main():
    args = parse_arguments()

    checkpoint_merger(
        primary_model=args.primary_model,
        output_path=args.output_path,
        secondary_model=args.secondary_model,
        tertiary_model=args.tertiary_model,
        interpolation=args.interpolation,
        multiplier=args.multiplier,
        half=args.half,
        discard_weights=args.discard_weights,
        validate=args.validate,
        prompt=args.prompt,
        image_output=args.image_output,
        sampler=args.sampler,
        sdxl=args.sdxl,
        disable_torch_compile=args.disable_torch_compile,
    )

if __name__ == "__main__":
    main()
    free_memory()
