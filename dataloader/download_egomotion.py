from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    filename="labels/egomotion/egomotion.chunk_0000.zip",
    repo_type="dataset",
    local_dir=".",
)
