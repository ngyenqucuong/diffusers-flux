import PIL
import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from insightface import FaceAnalysis
from insightface.model_zoo import init_recognition_model
from insightface.utils import face_align
from diffusers import FluxControlNetModel, FluxControlInpaintPipeline,AutoencoderKL
import torchvision.transforms as T
from PIL import Image
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPFeatureExtractor, CLIPModel
from huggingface_hub import hf_hub_download

import os

# Constants for RTX 4090 setup
INSIGHTFACE_DIR = "/insightface/models"
DEVICE = torch.device("cuda:0")  # Fixed GPU device
DTYPE = torch.bfloat16  # Optimal for RTX 4090

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=8,
        embedding_dim=512,
        output_dim=4096,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)

def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None):
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.to(DEVICE).contiguous()
    
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    
    face_emb = arcface_model(arc_face_image)[0]
    return face_emb

class InfiniteYou(torch.nn.Module):
    def __init__(self, adapter_model):
        super().__init__()
        self.image_proj_model = self.init_proj()
        self.image_proj_model.load_state_dict(adapter_model["image_proj"])
        
        # Load face encoder with multiple resolutions
        self.app_640 = FaceAnalysis(name='antelopev2', 
                                root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))
        
        self.app_320 = FaceAnalysis(name='antelopev2', 
                                root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))
        
        self.app_160 = FaceAnalysis(name='antelopev2', 
                                root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))
        
        self.arcface_model = init_recognition_model('arcface', device='cuda')

    def init_proj(self):
        return Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=8,
            embedding_dim=512,
            output_dim=4096,
            ff_mult=4
        )

    def detect_face(self, id_image_cv2):
        # Try different resolutions for better detection
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
            
        face_info = self.app_160.get(id_image_cv2)
        return face_info
    
    def get_face_embed_and_landmark(self, ref_image):
        id_image_cv2 = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
        face_info = self.detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')

        # Use largest face
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        landmark = face_info['kps']
        
        id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float().to(DEVICE)
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device=DEVICE, dtype=DTYPE)
        
        return id_embed, face_info['kps']
    
    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])
    
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    
    out_img = (out_img * 0.6).astype(np.uint8)
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
    
    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def tensor_to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [0, 1, 2]].numpy()
    return image

def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val
            n[1][k] = val
        c.append(n)
    return c







class FluxFaceSwapRTX4090:
    def __init__(self, adapter_path="./models/aes_stage2_img_proj.bin"):
        """
        Initialize FLUX Face Swap optimized for RTX 4090
        
        Args:
            adapter_path: Path to InfiniteYou adapter checkpoint
        """
        self.device = DEVICE
        self.dtype = DTYPE
        
        print(f"Initializing on {self.device} with dtype {self.dtype}")
        
        # Load InfiniteYou adapter
        print(f"Loading adapter from {adapter_path}")
        adapter_model_state_dict = torch.load(adapter_path, map_location="cpu")
        
        self.infinite_you = InfiniteYou(adapter_model_state_dict)
        self.infinite_you.to(self.device, dtype=self.dtype)
        
        print("InfiniteYou model loaded successfully!")

    def encode_prompt(self, clip, text):
        """Encode text prompt using CLIP"""
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None")
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    def prepare_condition_inpainting(self, positive, negative, pixels, vae, mask):
        """Prepare inpainting conditioning"""
        # Ensure dimensions are multiple of 8
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(pixels.shape[1], pixels.shape[2]), 
            mode="bilinear"
        )

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        # Apply mask to image
        m = (1.0 - mask.round()).squeeze(1).to(self.device, dtype=self.dtype)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        
        # Encode to latent space
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {
            "samples": orig_latent,
            "noise_mask": mask
        }

        out = []
        for conditioning in [positive, negative]:
            c = conditioning_set_values(conditioning, {
                "concat_latent_image": concat_latent,
                "concat_mask": mask
            })
            out.append(c)
        
        return out[0], out[1], out_latent

    def prepare_mask_and_landmark(self, image, blur_kernel=9):
        """Create face mask and extract landmarks"""
        image_detect = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_info = self.infinite_you.detect_face(image_detect)
        
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        
        # Use largest face
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        landmark = face_info['kps']

        # Create mask from face bbox
        mask = np.zeros((image.size[1], image.size[0]))
        x1, y1, x2, y2 = face_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Expand bbox by 1/3
        width = x2 - x1
        height = y2 - y1
        expand_x = width // 3
        expand_y = height // 3

        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(image.size[0], x2 + expand_x)
        new_y2 = min(image.size[1], y2 + expand_y)

        # Apply expanded mask and blur
        mask[new_y1:new_y2, new_x1:new_x2] = 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return mask, landmark

    def apply_face_swap(self, control_net, clip, ref_image, image, vae, 
                       weight=1.0, start_at=0.0, end_at=1.0, blur_kernel=9, 
                       noise=0.35, combine_embeds='average', mask=None):
        """
        Apply face swap using InfiniteYou
        
        Args:
            control_net: FLUX ControlNet model
            model: FLUX UNet model  
            clip: CLIP text encoder
            ref_image: Reference image tensor (face to use)
            image: Source image tensor (face to replace)
            vae: VAE model for encoding/decoding
            weight: ControlNet strength
            start_at: ControlNet start timestep
            end_at: ControlNet end timestep
            blur_kernel: Mask blur kernel size
            noise: Noise amount for unconditional embedding
            combine_embeds: How to combine multiple face embeddings
            mask: Optional face mask
        """
        
        # Convert tensors to PIL Images
        ref_image_pil = tensor_to_image(ref_image)
        ref_image_pil = PIL.Image.fromarray(ref_image_pil.astype(np.uint8)).convert("RGB")

        tensor_image = image.clone()
        image_pil = tensor_to_image(image)
        image_pil = PIL.Image.fromarray(image_pil.astype(np.uint8)).convert("RGB")

        # Prepare mask and landmarks
        if mask is None:
            mask, landmark = self.prepare_mask_and_landmark(image_pil, blur_kernel)
        else:
            _, landmark = self.infinite_you.get_face_embed_and_landmark(image_pil)

        # Prepare prompts
        prompt = " "
        neg_prompt = "ugly, blurry"
        positive = self.encode_prompt(clip, prompt)
        positive = conditioning_set_values(positive, {"guidance": float(1.5)})
        negative = self.encode_prompt(clip, neg_prompt)

        # Move tensor to device
        tensor_image = tensor_image.to(self.device, dtype=self.dtype)
        
        # Prepare inpainting conditioning
        positive, negative, latent_image = self.prepare_condition_inpainting(
            positive, negative, tensor_image, vae, mask
        )

        # Extract face embedding from reference
        face_embed, _ = self.infinite_you.get_face_embed_and_landmark(ref_image_pil)

        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        # Process face embeddings
        clip_embed = face_embed
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        # Create unconditional embedding
        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # Get image embeddings from InfiniteYou
        self.infinite_you = self.infinite_you.to(self.device, dtype=self.dtype)
        image_prompt_embeds, uncond_image_prompt_embeds = self.infinite_you.get_image_embeds(
            clip_embed.to(self.device, dtype=self.dtype), 
            clip_embed_zeroed.to(self.device, dtype=self.dtype)
        )

        # Create keypoint control image
        image_kps = draw_kps(image_pil, landmark)
        face_kps = torch.stack([T.ToTensor()(image_kps)], dim=0).permute([0,2,3,1])

        # Setup ControlNet conditioning
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(
                        face_kps.movedim(-1,1), weight, (start_at, end_at), vae=vae
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = (
                    image_prompt_embeds.to(DEVICE, dtype=c_net.cond_hint_original.dtype) if is_cond 
                    else uncond_image_prompt_embeds.to(DEVICE, dtype=c_net.cond_hint_original.dtype)
                )

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return cond_uncond[0], cond_uncond[1], latent_image


def preparebefore():

# Usage example for RTX 4090
if __name__ == "__main__":
    """
    Example usage on RTX 4090 with RunPod
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {DEVICE}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize face swapper
    face_swapper = FluxFaceSwapRTX4090(
        adapter_path="./models/aes_stage2_img_proj.bin"
    )
    
    print("Face swapper initialized successfully!")
    print("Ready for face swapping on RTX 4090!")
    
    vae_model_path = hf_hub_download('https://huggingface.co/frankjoshua/FLUX.1-dev/resolve/main/ae.safetensors')
    vae =  AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)  # Replace with your VAE model
    clip_model_path = hf_hub_download("")
    clip = CLIPModel.from_pretrained(clip_model_path, torch_dtype=torch.float16)  # Replace with your CLIP model
    infu_model_path = os.path.join('./models/InfiniteYou', f'infu_flux_v1.0', 'aes_stage2')
    infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
    control_net = FluxControlNetModel.from_pretrained(
        infusenet_path,
        torch_dtype=torch.bfloat16,
    )
    ref_image = 'path/to/your/reference/image.jpg'
    source_image = 'path/to/your/source/image.jpg'
    positive, negative, latent = face_swapper.apply_face_swap(
        control_net=control_net,
        clip=clip,
        ref_image=ref_image,
        image=source_image,
        vae=vae,
        weight=1.0,
        blur_kernel=9
    )
    print(positive)
    print(negative)
    print(latent)
    # transformer = FluxTransformer2DModel.from_pretrained("./models/black-forest-labs/FLUX.1-Depth-dev", subfolder="transformer", torch_dtype=torch.bfloat16)
    # text_encoder_2 = T5EncoderModel.from_pretrained("./models/black-forest-labs/FLUX.1-Depth-dev", subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    # pipe = FluxControlInpaintPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-Depth-dev",
    #     transformer=transformer,
    #     text_encoder_2=text_encoder_2,
    #     controlnet=control_net,
    #     torch_dtype=torch.bfloat16,
    # )
    # pipe.enable_model_cpu_offload()
    # pipe.to("cuda")
    
    # output = pipe(
    #     prompt=prompt,
    #     image=image,
    #     control_image=control_image,
    #     mask_image=mask_image,
    #     num_inference_steps=30,
    #     strength=0.9,
    #     guidance_scale=10.0,
    #     generator=torch.Generator().manual_seed(42),
    # ).images[0]

    # print("Face swap conditioning completed!")

    
