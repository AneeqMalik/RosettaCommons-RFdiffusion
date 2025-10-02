FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install --no-install-recommends -y \
		build-essential \
		ca-certificates \
		cmake \
		git \
		ninja-build \
		python3.9 \
		python3.9-dev \
		python3.9-distutils \
		python3.9-venv \
		python3-pip \
		wget \
	&& rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
	&& update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN python -m pip install --upgrade pip setuptools wheel

# RFdiffusion and its compiled extensions target NumPy 1.x.
RUN python -m pip install --no-cache-dir "numpy<2"

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN git clone --depth=1 https://github.com/RosettaCommons/RFdiffusion.git /opt/RFdiffusion

# Preload default RFdiffusion checkpoints into the SageMaker model directory.
RUN mkdir -p /opt/ml/model \
	&& bash /opt/RFdiffusion/scripts/download_models.sh /opt/ml/model

# Unpack optional PPI scaffold bundle so examples reference concrete files.
RUN tar -xzvf /opt/RFdiffusion/examples/ppi_scaffolds_subset.tar.gz -C /opt/RFdiffusion/examples

RUN python -m pip install --no-cache-dir dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
RUN python -m pip install --no-cache-dir torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python -m pip install --no-cache-dir \
	decorator==5.1.0 \
	e3nn==0.3.3 \
	hydra-core==1.3.2 \
	pynvml==11.0.0 \
	pyrsistent==0.19.3 \
	wandb==0.12.0

RUN python -m pip install --no-cache-dir -r /opt/RFdiffusion/env/SE3Transformer/requirements.txt \
	&& python -m pip install --no-cache-dir /opt/RFdiffusion/env/SE3Transformer

RUN python -m pip install --no-cache-dir /opt/RFdiffusion --no-deps

# Ensure example assets (e.g. default input PDBs and scaffolds) are available to the installed package.
RUN python - <<'PY'
import pathlib
import shutil
import site

source = pathlib.Path("/opt/RFdiffusion/examples")

targets = set()
targets.update(pathlib.Path(p) / "rfdiffusion" / "examples" for p in site.getsitepackages())
user_site = site.getusersitepackages()
if user_site:
	targets.add(pathlib.Path(user_site) / "rfdiffusion" / "examples")

for target in targets:
	if target.exists():
		shutil.rmtree(target)
	target.parent.mkdir(parents=True, exist_ok=True)
	shutil.copytree(source, target)
PY

COPY . .

ENV RF_DIFFUSION_HOME=/opt/RFdiffusion \
	SM_MODEL_DIR=/opt/ml/model \
	PYTHONUNBUFFERED=1 \
	DGLBACKEND=pytorch

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]