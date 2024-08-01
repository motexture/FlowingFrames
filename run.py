import torch
import random
import subprocess
import json
import numpy as np
import cv2
import os
import gradio as gr

from pipeline.flux_diff import FluxDiffPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import DPMSolverMultistepScheduler

def match_contrast_and_brightness(current_latents, previous_latents):
    prev_mean = previous_latents.mean(axis=(0, 2, 3, 4), keepdims=True)
    prev_std = previous_latents.std(axis=(0, 2, 3, 4), keepdims=True)

    current_mean = current_latents.mean(axis=(0, 2, 3, 4), keepdims=True)
    current_std = current_latents.std(axis=(0, 2, 3, 4), keepdims=True)

    adjusted_latents = (current_latents - current_mean) * (prev_std / (current_std + 1e-8)) + prev_mean
    adjusted_latents = adjusted_latents.clip(-1, 1)
    
    return adjusted_latents

def denormalize_to_image(normalized_tensor):
    normalized_tensor = normalized_tensor.squeeze(1)

    if normalized_tensor.is_cuda:
        normalized_tensor = normalized_tensor.cpu()
    
    if normalized_tensor.dim() == 5:
        normalized_tensor = normalized_tensor.squeeze(0)
        
    denormalized = (normalized_tensor + 1.0) * 127.5
    denormalized = torch.clamp(denormalized, 0, 255)
    
    uint8_numpy = denormalized.to(torch.uint8).numpy()
    uint8_numpy = np.squeeze(denormalized, axis=0)
    
    return uint8_numpy

def denormalize(normalized_tensor):
    if normalized_tensor.is_cuda:
        normalized_tensor = normalized_tensor.cpu()
    
    if normalized_tensor.dim() == 5:
        normalized_tensor = normalized_tensor.squeeze(0)
        
    denormalized = (normalized_tensor + 1.0) * 127.5
    denormalized = torch.clamp(denormalized, 0, 255)
    
    uint8_tensor = denormalized.to(torch.uint8)
    uint8_numpy = uint8_tensor.permute(1, 2, 3, 0).numpy()
    
    return uint8_numpy

def save_video(normalized_tensor, output_path, fps=30):
    denormalized_frames = denormalize(normalized_tensor)
    height, width = denormalized_frames.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in denormalized_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()

def set_torch_2_attn(unet):
    optim_count = 0
    
    for _, module in unet.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            for m in module:
                if isinstance(m, BasicTransformerBlock):
                    set_processors([m.attn1, m.attn2])
                    optim_count += 1

    print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def encode_video(input_file, output_file, height):
    command = ['ffmpeg',
               '-i', input_file,
               '-c:v', 'libx264',
               '-crf', '23',
               '-preset', 'fast',
               '-c:a', 'aac',
               '-b:a', '128k',
               '-movflags', '+faststart',
               '-vf', f'scale=-1:{height}',
               '-y',
               output_file]
    
    subprocess.run(command, check=True)

def get_video_height(input_file):
    command = ['ffprobe', 
               '-v', 'quiet', 
               '-print_format', 'json', 
               '-show_streams', 
               input_file]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)
    
    for stream in video_info.get('streams', []):
        if stream['codec_type'] == 'video':
            return stream['height']

    return None

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class VideoGenerator:
    def __init__(self):
        self.device = "cuda"
        self.stacked_latents = None
        self.previous_latents = None
        self.video_path = "outputs/concatenated_output.mp4"
        self.encoded_path = "outputs/concatenated_output_encoded.mp4"
        os.makedirs("outputs", exist_ok=True)

    def set_pipeline(self, model):
        self.pipeline = self.initialize_pipeline(model)

    def initialize_pipeline(self, model):  
        print("Loading pipeline...")
        
        pipeline = FluxDiffPipeline.from_pretrained(pretrained_model_name_or_path=model, use_safetensors=False).to(device=self.device, dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", algorithm_type="sde-dpmsolver++")
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.vae.enable_slicing()

        return pipeline

    def generate(self, prompt, negative_prompt, guidance_scale, reset, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, interpolation_strength):
        if self.seed != -1:
            set_seed(self.seed)

        with torch.no_grad(), torch.autocast(self.device):
            latents = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=self.device,
                previous_latents=self.previous_latents,
                interpolation_strength=interpolation_strength
            )
            
            self.stacked_latents = torch.cat((self.stacked_latents, latents), dim=2) if self.stacked_latents is not None else latents
            latents = match_contrast_and_brightness(latents, self.stacked_latents)
            self.previous_latents = latents[:, :, -num_conditioning_frames:, :, :]

            if reset:
                self.generate(prompt, negative_prompt, guidance_scale, False, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, interpolation_strength)

            save_video(self.decode(self.stacked_latents[:, :, num_frames + 1:, :, :]), self.video_path, fps)
            try:
                encode_video(self.video_path, self.encoded_path, get_video_height(self.video_path))
                os.remove(self.video_path)
            except:
                self.encoded_path = self.video_path
                pass

            return self.encoded_path
        
    def decode(self, latents):
        latents = 1 / self.pipeline.vae.config.scaling_factor * latents

        batch, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, channels, height, width)
        image = self.pipeline.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )

        return video.float()        
        
    def reset_and_generate_initial(self, prompt, negative_prompt, guidance_scale, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, seed, interpolation_strength):
        self.stacked_latents = None
        self.previous_latents = None
        self.seed = seed

        return self.generate(prompt, negative_prompt, guidance_scale, True, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, interpolation_strength)

video_gen = VideoGenerator()
initial_generated = False

with gr.Blocks() as iface:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>FluxDiff</h1>
        <p>A text-to-video model that uses past frames for conditioning, enabling the generation of infinite-length videos.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value="Darth Vader is surfing on the ocean")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=30.0, step=0.1, value=12)

            gr.Markdown("## Model Information")
            gr.Markdown("<small>Recommended: 1024x576 resolution. Other resolutions might work as well, however they require testing.</small>")

            width = gr.Slider(label="Width", minimum=64, maximum=1280, step=64, value=1024)
            height = gr.Slider(label="Height", minimum=64, maximum=1280, step=64, value=576)

            gr.Markdown("## Frame Settings")
            gr.Markdown("<small>Recommended: 4 frames & 4 conditioning frames, or 6 frames & 4 conditioning frames. Other settings may need testing.</small>")

            num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=24, step=1, value=6)
            num_conditioning_frames = gr.Slider(label="Number of Conditioning Frames", minimum=1, maximum=8, step=1, value=4)

            gr.Markdown("## Inference Settings")
            gr.Markdown("<small>More inference steps generally increase motion. Adjust as needed.</small>")

            num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=30)
            fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=6)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=1000000, step=1, value=-1)

            gr.Markdown("## Interpolation")
            gr.Markdown("<small>Interpolation 0.0: full strength to previous prompt. 1.0: full strength to new prompt.</small>")

            interpolation_strength = gr.Slider(label="Interpolation Strength", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        
        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Video")
            generate_initial_button = gr.Button("Generate Initial Video")
            extend_button = gr.Button("Extend Video", interactive=False)

        def on_generate_initial(prompt, negative_prompt, guidance_scale, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, seed, interpolation_strength):
            global initial_generated
            video_path = video_gen.reset_and_generate_initial(prompt, negative_prompt, guidance_scale, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, seed, interpolation_strength)
            initial_generated = True
            return video_path

        def on_extend(prompt, negative_prompt, guidance_scale, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, interpolation_strength):
            if initial_generated:
                return video_gen.generate(prompt, negative_prompt, guidance_scale, False, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, interpolation_strength)
            return None

        generate_initial_button.click(
            on_generate_initial, 
            inputs=[prompt, negative_prompt, guidance_scale, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, seed, interpolation_strength],
            outputs=video_output
        ).then(
            lambda: gr.update(interactive=True), 
            outputs=extend_button
        )

        extend_button.click(
            on_extend, 
            inputs=[prompt, negative_prompt, guidance_scale, width, height, num_frames, num_conditioning_frames, num_inference_steps, fps, interpolation_strength],
            outputs=video_output
        )

if __name__ == "__main__":
    video_gen.set_pipeline("motexture/FluxDiff")
    iface.launch()
