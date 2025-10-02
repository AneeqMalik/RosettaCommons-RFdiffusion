# RFdiffusion SageMaker Inference Image

This repository wraps the [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) project in a GPU-enabled Flask/Gunicorn service that is compatible with Amazon SageMaker custom inference containers. Incoming requests are bridged to the original `scripts/run_inference.py` entrypoint via Python's `subprocess` module, and all generated artefacts are mirrored into the SageMaker output directory.

## Features

- 🧠 **Native RFdiffusion**: The Docker image clones the upstream repository, installs NVIDIA's SE(3) Transformer, and keeps all Hydra configuration files intact.
- ⚙️ **Configurable runtime**: Control contigs, Hydra overrides, GPU visibility, and output handling through the request payload.
- 📦 **SageMaker friendly**: Uses the standard `/ping` and `/invocations` endpoints, honours `SM_MODEL_DIR` and `SM_OUTPUT_DATA_DIR`, and returns structured JSON responses.
- ✅ **Test coverage**: Lightweight unit tests ensure the HTTP layer builds commands correctly and captures output artefacts.

## Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU access.
- AWS ECR permissions if you intend to host the image on SageMaker.
- Pre-downloaded RFdiffusion checkpoint weights (or enable the auto-download flag at runtime).

## Build the container

```bash
# clone this repository and change into the directory first

docker build -t rfdiffusion-sagemaker:latest .
```

To run locally with GPU acceleration:

```bash
mkdir -p $PWD/models $PWD/outputs
# Optionally download model weights ahead of time
docker run --rm --gpus all \
  -e SM_MODEL_DIR=/opt/ml/model \
  -v $PWD/models:/opt/ml/model \
  -v $PWD/outputs:/opt/ml/output/data \
  -p 8080:8080 \
  rfdiffusion-sagemaker:latest
```

## Upload to Amazon ECR (optional)

```bash
aws ecr create-repository --repository-name rfdiffusion-sagemaker || true
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag rfdiffusion-sagemaker:latest <account>.dkr.ecr.<region>.amazonaws.com/rfdiffusion-sagemaker:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/rfdiffusion-sagemaker:latest
```

## SageMaker model data layout

SageMaker mounts the model artefact tarball into `SM_MODEL_DIR` (default `/opt/ml/model`). Place the RFdiffusion `.pt` checkpoint files in this directory. At runtime you can request automatic downloads by sending `"auto_download_weights": true` in the invocation payload (requires outbound internet access) or by preprocessing the model tarball with the checkpoints already included.

## Service endpoints

- `GET /ping` – returns `"pong"` for health checks.
- `POST /invocations` – executes RFdiffusion and returns a JSON payload.

Sample request body:

```json
{
  "contigs": "[150-150]",
  "num_designs": 2,
  "run_name": "demo",
  "hydra_overrides": ["diffuser.T=20"],
  "inline_outputs": false,
  "auto_download_weights": false
}
```

Response excerpt:

```json
{
  "status": "ok",
  "duration_seconds": 123.4,
  "outputs": [
    {
      "filename": "demo_0.pdb",
      "path": "/tmp/rfdiffusion/demo_0.pdb"
    }
  ]
}
```

Set `inline_outputs` to `true` if you want base64-encoded artefacts embedded directly in the response. When a SageMaker endpoint is used, all generated files are also copied to `SM_OUTPUT_DATA_DIR`.

## Invocation tips

- Provide additional Hydra overrides through `hydra_overrides` (list of strings), e.g. `"contigmap.inpaint_seq=[10-20]"`.
- Override the model directory with `"model_directory": "/opt/ml/model"` if your weights live elsewhere.
- Control GPU affinity by passing `"cuda_visible_devices": "0"`.
- Use `"timeout_seconds"` to cap long-running jobs (default 7200 seconds).

## Running tests

```bash
python -m unittest
```

## Troubleshooting

- **Weights missing**: Ensure `.pt` checkpoint files are present under `SM_MODEL_DIR` or enable auto download.
- **CUDA errors**: Confirm your host exposes GPUs to the container (`docker run --gpus all`) and that compatible NVIDIA drivers are installed.
- **Hydra override issues**: Each override should be a single string exactly as you would type it on the CLI (no enclosing quotes needed when using the JSON API).

Happy diffusing! 🚀
