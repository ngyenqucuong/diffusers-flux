import torch
import random
import numpy as np
from PIL import Image

from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from pipeline import InstantCharacterFluxPipeline

from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid
import json
import io
import uvicorn

# global variable

birefnet_transform_image = None
pipe = None
birefnet  = None
executor = ThreadPoolExecutor(max_workers=1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory job storage
jobs = {}
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def initialize_pipelines():
    """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
    global pipe, birefnet_transform_image,birefnet
    
    try:
        ip_adapter_path = hf_hub_download(repo_id="Tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
        base_model = 'black-forest-labs/FLUX.1-dev'
        image_encoder_path = 'google/siglip-so400m-patch14-384'
        image_encoder_2_path = 'facebook/dinov2-giant'
        birefnet_path = 'ZhengPeng7/BiRefNet'

        pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        # load InstantCharacter
        pipe.init_adapter(
            image_encoder_path=image_encoder_path, 
            image_encoder_2_path=image_encoder_2_path, 
            subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
        )

        # load matting model
        birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
        birefnet.to('cuda')
        birefnet.eval()
        birefnet_transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        raise



def remove_bkg(subject_image):

    def infer_matting(img_pil):
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to('cuda')

        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None]
        return mask

    def get_bbox_from_mask(mask, th=128):
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1

        sample = np.max(mask, axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x1 = idx
                break
        
        sample = np.max(mask[:, ::-1], axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x2 = width - 1 - idx
                break

        sample = np.max(mask, axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y1 = idx
                break

        sample = np.max(mask[::-1], axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y2 = height - 1 - idx
                break

        x1 = np.clip(x1, 0, width-1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height-1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width-1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height-1).round().astype(np.int32)

        return [x1, y1, x2, y2]

    def pad_to_square(image, pad_value = 255, random = False):
        '''
            image: np.array [h, w, 3]
        '''
        H,W = image.shape[0], image.shape[1]
        if H == W:
            return image

        padd = abs(H - W)
        if random:
            padd_1 = int(np.random.randint(0,padd))
        else:
            padd_1 = int(padd / 2)
        padd_2 = padd - padd_1

        if H > W:
            pad_param = ((0,0),(padd_1,padd_2),(0,0))
        else:
            pad_param = ((padd_1,padd_2),(0,0),(0,0))

        image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
        return image

    salient_object_mask = infer_matting(subject_image)[..., 0]
    x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)
    subject_image = np.array(subject_image)
    salient_object_mask[salient_object_mask > 128] = 255
    salient_object_mask[salient_object_mask < 128] = 0
    sample_mask = np.concatenate([salient_object_mask[..., None]]*3, axis=2)
    obj_image = sample_mask / 255 * subject_image + (1 - sample_mask / 255) * 255
    crop_obj_image = obj_image[y1:y2, x1:x2]
    crop_pad_obj_image = pad_to_square(crop_obj_image, 255)
    subject_image = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
    return subject_image



def create_image(input_image,
                 prompt,
                 scale, 
                 guidance_scale,
                 num_inference_steps,
                 width,
                 height,
                 seed
                 ):
    
    input_image = remove_bkg(input_image)

    images = pipe(
        prompt=prompt, 
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        subject_image=input_image,
        subject_scale=scale,
        generator=torch.manual_seed(seed),
        num_samples=1
    ).images

    
    return images



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
    width: int = 512
    height: int = 512
    ip_adapter_scale: float = 0.8  # Lower for InstantID
    controlnet_conditioning_scale: float = 0.8
    guidance_scale: float = 0.0  # Zero for LCM
    num_inference_steps: int = 8

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None



async def gen_img2img(job_id: str, face_image : Image.Image,pose_image: Image.Image,request: Img2ImgRequest):
    
    seed = request.seed if request.seed else  random.randint(0, np.iinfo(np.int32).max)
    image = create_image(face_image,
                         request.prompt ,
                         scale=request.ip_adapter_scale,
                         guidance_scale=request.guidance_scale,
                         num_inference_steps=request.num_inference_steps,
                         width=request.width,
                         height=request.height,
                         seed=seed)[0]
    filename = f"{job_id}_base.png"
    filepath = os.path.join(results_dir, filename)
    image.save(filepath)
        
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
        with open("interface.html", "r") as f:
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
        "gpu_info": gpu_info
    }


@app.post("/img2img")
async def img2img(
    base_image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form("(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"),
    strength: float = Form(0.85),
    ip_adapter_scale: float = Form(0.8),  # Lower for InstantID
    width: int = Form(512),
    height: int = Form(512),
    controlnet_conditioning_scale: float = Form(0.8),
    num_inference_steps: int = Form(8),
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
        base_img = Image.open(io.BytesIO(await base_image.read())).convert('RGB')
        request = Img2ImgRequest(
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
           
        )
        # Start background task
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, lambda: asyncio.run(
            gen_img2img(job_id, base_img, pose_img, request)
        ))
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        raise HTTPException(status_code=400, detail=str(e))


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
    
    # Set environment variables for better CUDA error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    uvicorn.run(app, host="0.0.0.0", port=8888)