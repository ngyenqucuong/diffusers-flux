import torch
from diffusers import FluxControlInpaintPipeline,AutoencoderKL,FluxControlNetModel
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import T5EncoderModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor  # https://github.com/huggingface/image_gen_aux
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
# import cv2
import logging
import json



executor = ThreadPoolExecutor(max_workers=1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
jobs = {}
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


def initialize_pipelines():
    """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
    global pipe
    
    vae_model_path = hf_hub_download('https://huggingface.co/frankjoshua/FLUX.1-dev/resolve/main/ae.safetensors')
    vae =  AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    control_net = FluxControlNetModel.from_pretrained(
        hf_hub_download(
        repo_id="vuongminhkhoi4/ComfyUI_InfiniteYou", subfolder="aes_stage2_control_net", filename="aes_stage2_control.safetensors"
    ),
        torch_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
    repo_name = "ByteDance/Hyper-SD"

    pipe = FluxControlInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev",
        vae=vae,
        control_net=control_net,
        torch_dtype=torch.bfloat16,
    )
    # use following lines if you have GPU constraints
    # ---------------------------------------------------------------


    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora(lora_scale=0.125)

    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2
    pipe.enable_model_cpu_offload()
    # ---------------------------------------------------------------
    pipe.to("cuda")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines on startup"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, initialize_pipelines)
    yield


app = FastAPI(title="SDXL Face Swap API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    seed: Optional[int] = None
    strength: float = 0.8
    ip_adapter_scale: float = 0.8  # Lower for InstantID
    controlnet_conditioning_scale: float = 0.8
    guidance_scale: float = 0.0  # Zero for LCM
    detail_face: bool = False  # Whether to refine face details
    num_inference_steps: int = 50  # Number of inference steps

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None



async def gen_img2img(job_id: str, face_image : Image.Image,pose_image: Image.Image,mask_image: Image.Image,request: Img2ImgRequest):
    negative_prompt = f"{request.negative_prompt}, blue artifacts, color bleeding, unnatural colors, mask edges, visible seams, hair"
    seed = request.seed if request.seed else torch.randint(0, 2**32, (1,)).item()
    # cv2image = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    prompt = "a blue robot singing opera with human-like expressions"
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

    head_mask = np.zeros_like(image)
    head_mask[65:580, 300:642] = 255
    mask_image = Image.fromarray(head_mask)

    processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    control_image = processor(image)[0].convert("RGB")

    generated_image = pipe(
        num_samples=1,
        prompt=prompt,
        # negative_prompt=negative_prompt,
        image=image,
        control_image=control_image,
        mask_image=mask_image,
        num_inference_steps=8,
        strength=0.9,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    
    

    # if request.detail_face:
    #     generated_image = detail_face(generated_image, face_image)
    filename = f"{job_id}_base.png"
    filepath = os.path.join(results_dir, filename)
    generated_image.save(filepath)
        
    metadata = {
        "job_id": job_id,
        "type": "head_swap",
        "seed": seed,
        "prompt": request.prompt,
        "parameters": request.dict(),
        "filename": filename,
        "device_used": 'cuda',
    }
        
    metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["progress"] = 1.0
    jobs[job_id]["result_url"] = f"/results/{filename}"
    jobs[job_id]["metadata"] = metadata
    jobs[job_id]["completed_at"] = datetime.now()
    
    logger.info(f"Img2img completed successfully on cuda")






@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    try:
        with open("img2img_interface.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1>")

@app.get("/web", response_class=HTMLResponse)
async def serve_web_interface_alt():
    """Alternative route for web interface"""
    return await serve_web_interface()

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_cached": torch.cuda.memory_reserved()
        }
    
    pipeline_device = None
    if pipe is not None:
        try:
            # Try to get device from unet (most reliable)
            pipeline_device = str(pipe.unet.device)
        except:
            try:
                # Fallback to vae device
                pipeline_device = str(pipe.vae.device)
            except:
                pipeline_device = "unknown"
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "pipeline_device": pipeline_device,
        "pipelines_loaded": pipe is not None,
        # "face_analysis_loaded": face_analysis_app is not None,
        "gpu_info": gpu_info
    }


@app.post("/img2img")
async def img2img(
    base_image: UploadFile = File(...),
    pose_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form("(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"),
    strength: float = Form(0.85),
    ip_adapter_scale: float = Form(0.8),  # Lower for InstantID
    controlnet_conditioning_scale: float = Form(0.8),
    num_inference_steps: int = Form(50),  # Number of inference steps
    detail_face: bool = Form(False),
    guidance_scale: float = Form(0),  # Zero for LCM
    seed: Optional[int] = Form(None),
    
):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "head_swap"
    }
    try:
    # Load images
        base_img = Image.open(io.BytesIO(await base_image.read())).resize((256, 256))
        pose_img = Image.open(io.BytesIO(await pose_image.read())).resize((512, 768))
        mask_img = Image.open(io.BytesIO(await mask_image.read())).resize((512, 768))
        request = Img2ImgRequest(

            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            detail_face=detail_face,
            num_inference_steps=num_inference_steps
            
        )
        # Start background task
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, lambda: asyncio.run(
            gen_img2img(job_id, base_img, pose_img, mask_img, request)
        ))
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Error processing img2img request: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        return {"job_id": job_id, "status": "failed", "error_message": str(e)}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result_url": job.get("result_url"),
        "seed": job.get("metadata", {}).get("seed"),
        "error_message": job.get("error_message"),
        "created_at": job["created_at"].isoformat(),
        "completed_at": job.get("completed_at").isoformat() if job.get("completed_at") else None
    }

@app.get("/results/{filename}")
async def get_result(filename: str):
    """Get result image"""
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath)


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    try:
        job_list = []
        for job_id, job_data in jobs.items():
            job_list.append({
                "job_id": job_id,
                "status": job_data.get("status", "unknown"),
                "created_at": job_data.get("created_at", datetime.now()).isoformat(),
                "completed_at": job_data.get("completed_at").isoformat() if job_data.get("completed_at") else None,
                "result_url": job_data.get("result_url"),
                "error_message": job_data.get("error_message")
            })
        
        job_list.sort(key=lambda x: x["created_at"], reverse=True)
        return job_list
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return []

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job = jobs[job_id]
    if "metadata" in job and "filename" in job["metadata"]:
        filename = job["metadata"]["filename"]
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete metadata file
        metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    
    # Set environment variables for better CUDA error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    uvicorn.run(app, host="0.0.0.0", port=8888)