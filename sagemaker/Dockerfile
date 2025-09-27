FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Install Python + AWS SageMaker requirements
RUN apt-get update && apt-get install -y python3 python3-pip docker.io
RUN pip3 install sagemaker-training

# Copy inference script
COPY . /opt/ml/model/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/ml/model/code:${PATH}"

ENTRYPOINT ["python3", "/opt/ml/model/code/inference.py"]
