FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Install deps
RUN apt-get update && apt-get install -y git python3 python3-pip

# Install RFdiffusion
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git /app/RFdiffusion
WORKDIR /app/RFdiffusion
RUN pip3 install -r requirements.txt

# Copy inference script
COPY inference.py /opt/ml/model/code/inference.py

# SageMaker expects entrypoint under /opt/ml/model/code
ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/ml/model/code:${PATH}"

ENTRYPOINT ["python3", "/opt/ml/model/code/inference.py"]
