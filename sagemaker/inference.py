import os
import subprocess
import tarfile

def input_fn(request_body, content_type):
    """Save input PDB file from S3 → container"""
    input_dir = "/opt/ml/input"
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, "input.pdb")

    with open(input_path, "wb") as f:
        f.write(request_body)

    return input_path


def predict_fn(input_data, model):
    """Run RFdiffusion inside the container"""
    output_dir = "/opt/ml/output"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python3", "scripts/run_inference.py",  # RFdiffusion entry
        f"inference.output_prefix={output_dir}/motifscaffolding",
        "inference.model_directory_path=/app/RFdiffusion/models",
        f"inference.input_pdb={input_data}",
        "inference.num_designs=3",
        "contigmap.contigs=[10-40/A20-40/10-40]"
    ]

    subprocess.run(cmd, check=True)
    return output_dir


def output_fn(prediction, accept):
    """Tarball all output files so SageMaker can store them"""
    tar_path = "/opt/ml/output/result.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(prediction, arcname=os.path.basename(prediction))
    with open(tar_path, "rb") as f:
        return f.read()
