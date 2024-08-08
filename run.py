import torch
import random
import numpy as np
import cv2
import os
import gradio as gr

from pipeline.flowing_frames import FlowingFramesPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import DPMSolverMultistepScheduler

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
        self.video_path = "outputs/output.mp4"
        os.makedirs("outputs", exist_ok=True)

    def set_pipeline(self, model):
        self.pipeline = self.initialize_pipeline(model)

    def initialize_pipeline(self, model):  
        print("Loading pipeline...")
        
        pipeline = FlowingFramesPipeline.from_pretrained(pretrained_model_name_or_path=model, use_safetensors=False).to(device=self.device, dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", algorithm_type="sde-dpmsolver++")
        pipeline.vae.enable_slicing()

        return pipeline

    def scale_latents_to_range(self, latents, min_val=-1, max_val=1):
        min_latent = latents.min()
        max_latent = latents.max()
        scaled_latents = (latents - min_latent) / (max_latent - min_latent) * (max_val - min_val) + min_val
        return scaled_latents

    def generate(self, prompt, negative_prompt, guidance_scale, reset, width, height, num_inference_steps, fps, interpolation_strength):
        if self.seed != -1:
            set_seed(self.seed)

        with torch.no_grad(), torch.autocast(self.device):
            latents = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=6,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=self.device,
                previous_latents=self.previous_latents,
                interpolation_strength=interpolation_strength
            )

            if self.stacked_latents is not None:
                self.stacked_latents = torch.cat((self.stacked_latents, latents), dim=2)
                if self.stacked_latents.size(2) > 6 * 2:
                    self.stacked_latents[:, :, -6 * 2:, :, :] = self.pipeline.refiner(self.stacked_latents[:, :, -6 * 2:, :, :])
            else:
                self.stacked_latents = latents
            self.previous_latents = self.stacked_latents[:, :, -4:, :, :]

            if reset:
                self.generate(prompt, negative_prompt, guidance_scale, False, width, height, num_inference_steps, fps, interpolation_strength)
            
            save_video(self.decode(self.stacked_latents[:, :, 7:, :, :]), self.video_path, fps)

            return self.video_path
        
    def normalize_latents(self, latents):
        mean = latents.mean(dim=[2, 3, 4], keepdim=True)
        std = latents.std(dim=[2, 3, 4], keepdim=True)
        normalized_latents = (latents - mean) / (std + 1e-5)
        return normalized_latents
        
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
        
    def reset_and_generate_initial(self, prompt, negative_prompt, guidance_scale, width, height, num_inference_steps, fps, seed, interpolation_strength):
        self.stacked_latents = None
        self.previous_latents = None
        self.seed = seed

        return self.generate(prompt, negative_prompt, guidance_scale, True, width, height, num_inference_steps, fps, interpolation_strength)

video_gen = VideoGenerator()
initial_generated = False

with gr.Blocks() as iface:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>FlowingFrames</h1>
        <p>A text-to-video model that uses past frames for conditioning, enabling the generation of infinite-length videos.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value="A photo of a beautiful red head girl with blue eyes")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, grainy, bad textures")
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=30.0, step=0.1, value=12)

            gr.Markdown("## Resolution Settings")

            width = gr.Slider(label="Width", minimum=576, maximum=1280, step=64, value=1024)
            height = gr.Slider(label="Height", minimum=576, maximum=1280, step=64, value=576)

            gr.Markdown("## Inference Settings")

            num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=30)
            fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=6)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=1000000, step=1, value=-1)

            gr.Markdown("## Interpolation")

            interpolation_strength = gr.Slider(label="Interpolation Strength", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        
        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Video")
            generate_initial_button = gr.Button("Generate Initial Video")
            extend_button = gr.Button("Extend Current Video", interactive=False)

        def on_generate_initial(prompt, negative_prompt, guidance_scale, width, height, num_inference_steps, fps, seed, interpolation_strength):
            global initial_generated
            video_path = video_gen.reset_and_generate_initial(prompt, negative_prompt, guidance_scale, width, height, num_inference_steps, fps, seed, interpolation_strength)
            initial_generated = True
            return video_path

        def on_extend(prompt, negative_prompt, guidance_scale, width, height, num_inference_steps, fps, interpolation_strength):
            if initial_generated:
                return video_gen.generate(prompt, negative_prompt, guidance_scale, False, width, height, num_inference_steps, fps, interpolation_strength)
            return None

        generate_initial_button.click(
            on_generate_initial,
            inputs=[prompt, negative_prompt, guidance_scale, width, height, num_inference_steps, fps, seed, interpolation_strength],
            outputs=video_output
        ).then(
            lambda: gr.update(interactive=True), 
            outputs=extend_button
        )

        extend_button.click(
            on_extend,
            inputs=[prompt, negative_prompt, guidance_scale, width, height, num_inference_steps, fps, interpolation_strength],
            outputs=video_output
        )

if __name__ == "__main__":
    video_gen.set_pipeline("motexture/FlowingFrames")
    iface.launch()
