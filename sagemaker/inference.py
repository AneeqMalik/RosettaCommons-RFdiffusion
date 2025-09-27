import os
import subprocess

def input_fn(request_body, content_type):
    # Save input PDB file to a known location
    input_path = "/opt/ml/input/input.pdb"
    with open(input_path, "wb") as f:
        f.write(request_body)
    return input_path

def predict_fn(input_data, model):
    # Run RFdiffusion with same args as your docker run
    output_path = "/opt/ml/output/motifscaffolding"
    os.makedirs("/opt/ml/output", exist_ok=True)

    cmd = [
        "python3", "run_inference.py",  # Or the RFdiffusion entry script
        f"inference.output_prefix={output_path}",
        "inference.model_directory_path=/app/RFdiffusion/models",
        f"inference.input_pdb={input_data}",
        "inference.num_designs=3",
        "contigmap.contigs=[10-40/A20-40/10-40]"
    ]
    subprocess.run(cmd, check=True)

    return output_path

def output_fn(prediction, accept):
    # Return the generated files (e.g., tar them up)
    import tarfile
    tar_path = "/opt/ml/output/result.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(prediction, arcname=os.path.basename(prediction))
    with open(tar_path, "rb") as f:
        return f.read()
