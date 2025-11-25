import os
import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Any, List


num_images_to_process = 10**6
num_gpus = 1

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_path = f"/tmp/shared_storage/process_images_output/{timestamp}"


def download_single_image(url: str, session: requests.Session) -> Dict[str, Any]:
    """Download a single image."""
    try:
        # Use the provided session for connection pooling
        response = session.get(url, timeout=5, stream=True)

        if response.status_code == 200:
            # Read content
            content = response.content
            return {"content": content, "status": "success", "url": url}
        else:
            # Return HTTP status code for non-200 responses
            return {
                "content": None,
                "status": f"http_{response.status_code}",
                "url": url,
            }

    except Exception as e:
        return {"content": None, "status": f"error_{type(e).__name__}", "url": url}


def image_download(batch: Dict[str, List]) -> Dict[str, List]:
    """Download a batch of images using a thread pool for parallelism."""
    urls = batch["url"]

    # Create a session for connection pooling
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100,
        pool_maxsize=100,
        max_retries=0,  # No automatic retries
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(
            executor.map(lambda url: download_single_image(url, session), urls)
        )

    batch["bytes"] = [r["content"] for r in results]
    return batch


def process_single_image(row: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single image: validate, convert to RGB, and resize."""
    image_bytes = row["bytes"]
    if image_bytes is None:
        return row

    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=95)
        row["bytes"] = output_buffer.getvalue()
    except Exception:
        row["bytes"] = None
    return row


vision_processor_config = vLLMEngineProcessorConfig(
    model_source="unsloth/Llama-3.1-8B-Instruct",
    concurrency=num_gpus,  # 1 vLLM engine replica
    batch_size=64,  # 32 samples per batch
    engine_kwargs={
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4096,  # Reduce if CUDA OOM occurs
        "max_model_len": 4096,  # Constrain to fit test GPU memory
    },
)


def vision_preprocess(row):
    image_bytes = row["bytes"]
    return dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(BytesIO(image_bytes)),
                    },
                ],
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=150,
            detokenize=False,
        ),
    )


def vision_postprocess(row):
    row.pop("bytes")
    return row


vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)

dataset = (
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        columns=["url"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=10,
        ray_remote_args={"memory": int(4 * 10**9)},
    )
    .limit(num_images_to_process)
    .repartition(target_num_rows_per_block=1000)
    .map_batches(
        image_download,
        batch_size=100,
        memory=(10**9),
    )
    .drop_columns(["url"])
    .map(process_single_image)
    .filter(
        lambda row: row["bytes"] is not None
    )  # Filter out failed downloads/processing
)

dataset = vision_processor(dataset)

dataset.write_parquet(output_path)
