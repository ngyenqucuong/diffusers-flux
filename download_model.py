from huggingface_hub import hf_hub_download, snapshot_download
import shutil
import os

#Download Image Projection models
os.makedirs("../models/InfiniteYou", exist_ok=True)

hf_hub_download(
    repo_id="vuongminhkhoi4/ComfyUI_InfiniteYou", subfolder="aes_stage2_control_net", filename="aes_stage2_img_proj.bin", local_dir="../models/InfiniteYou"
)
hf_hub_download(
    repo_id="vuongminhkhoi4/ComfyUI_InfiniteYou", subfolder="sim_stage1_control_net", filename="sim_stage1_img_proj.bin", local_dir="../models/InfiniteYou"
)

#Download ControlNet models

hf_hub_download(
    repo_id="vuongminhkhoi4/ComfyUI_InfiniteYou", subfolder="sim_stage1_control_net", filename="sim_stage1_control_net.safetensors", local_dir="../models/controlnet"
)

#Download InsightFace model
os.makedirs("../models/insightface", exist_ok=True)

snapshot_download(
    repo_id="vuongminhkhoi4/antelopev2",
    cache_dir ="models",
    repo_type ="model",
    local_dir="../models/insightface",
)
if os.path.exists("./models") and os.path.isdir("./models"):
    shutil.rmtree("./models")