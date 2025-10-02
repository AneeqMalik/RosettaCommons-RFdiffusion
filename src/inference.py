import base64
import glob
import logging
import os
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List


LOGGER = logging.getLogger(__name__)


def load_model(model_dir: str) -> Dict[str, Any]:
    """Prepare contextual metadata required to invoke RFdiffusion.

    The SageMaker training/inference toolkit mounts the model artifact into the
    directory specified via ``SM_MODEL_DIR``. We do not load any model into
    memory here. Instead we validate the RFdiffusion installation location and
    prepare a context dictionary consumed by :func:`predict`.
    """

    repo_path = os.environ.get("RF_DIFFUSION_HOME", "/opt/RFdiffusion")
    script_path = os.path.join(repo_path, "scripts", "run_inference.py")

    if not os.path.exists(script_path):
        raise FileNotFoundError(
            "RFdiffusion inference script not found at expected location.",
            script_path,
        )

    os.makedirs(model_dir, exist_ok=True)

    python_executable = os.environ.get("RF_PYTHON_EXECUTABLE", "python3")

    context: Dict[str, Any] = {
        "repo_path": repo_path,
        "script_path": script_path,
        "model_dir": model_dir,
        "python_executable": python_executable,
        "download_script": os.path.join(repo_path, "scripts", "download_models.sh"),
    }

    LOGGER.info("RFdiffusion context prepared: repo=%s", repo_path)
    return context


def predict(body: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute RFdiffusion via subprocess and marshal the response.

    Parameters
    ----------
    body:
        JSON payload provided by the caller. Supported keys:

        - ``contigs`` (str): Hydrated string for ``contigmap.contigs``.
        - ``output_prefix`` (str): Absolute path prefix for design artefacts.
        - ``output_dir`` (str): Base directory used when ``output_prefix`` isn't provided.
        - ``run_name`` (str): Friendly identifier embedded in generated filenames.
        - ``num_designs`` (int): Number of designs to generate (defaults to ``1``).
        - ``config_name`` (str): Optional Hydra config name supplied via ``--config-name``.
        - ``model_directory`` (str): Override for RFdiffusion checkpoint directory.
        - ``input_pdb`` (str): Optional input PDB path override (forwarded as-is).
        - ``hydra_overrides`` (list[str]): Additional hydra command-line overrides.
        - ``inline_outputs`` (bool): When ``True`` embed generated artefacts as base64.
        - ``timeout_seconds`` (int): Execution timeout (defaults to ``RF_TIMEOUT_SECONDS`` env or 7200s).
        - ``cuda_visible_devices`` (str): Override ``CUDA_VISIBLE_DEVICES`` for the subprocess.
        - ``auto_download_weights`` (bool): Fetch model weights using ``download_models.sh`` when none are present.

    context:
        Context object emitted by :func:`load_model`.
    """

    if body is None:
        raise ValueError("Request payload must be a JSON object.")

    if not isinstance(body, dict):
        raise TypeError("Request payload must be a JSON dictionary.")

    hydra_overrides = body.get("hydra_overrides", [])
    if hydra_overrides and not isinstance(hydra_overrides, list):
        raise TypeError("'hydra_overrides' must be provided as a list of strings")

    inline_outputs = bool(body.get("inline_outputs", False))

    run_name = body.get("run_name") or f"design_{uuid.uuid4().hex[:8]}"

    output_prefix = body.get("output_prefix")
    if output_prefix:
        output_base = os.path.dirname(output_prefix)
    else:
        output_base = body.get("output_dir") or os.environ.get("RF_OUTPUT_DIR", "/tmp/rfdiffusion")
        output_prefix = os.path.join(output_base, run_name)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    model_directory = body.get("model_directory") or context["model_dir"]
    os.makedirs(model_directory, exist_ok=True)

    _maybe_download_weights(model_directory, context, body)

    num_designs = int(body.get("num_designs", 1))
    config_name = body.get("config_name")
    contigs = body.get("contigs")
    input_pdb = body.get("input_pdb")
    timeout_seconds = int(
        body.get("timeout_seconds")
        or os.environ.get("RF_TIMEOUT_SECONDS")
        or 7200
    )

    command: List[str] = [context["python_executable"], context["script_path"]]
    if config_name:
        command.append(f"--config-name={config_name}")

    command.extend(
        [
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={num_designs}",
            f"inference.model_directory_path={model_directory}",
        ]
    )

    if contigs:
        command.append(f"contigmap.contigs={contigs}")

    if input_pdb:
        command.append(f"inference.input_pdb={input_pdb}")

    for override in hydra_overrides:
        command.append(str(override))

    environment = os.environ.copy()
    cuda_visible_devices = body.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    start_time = time.time()
    LOGGER.info("Launching RFdiffusion: %s", command)

    result = subprocess.run(
        command,
        cwd=context["repo_path"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )

    if result.returncode != 0:
        LOGGER.error("RFdiffusion run failed: %s", result.stderr)
        raise RuntimeError(
            "RFdiffusion execution failed",
            {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
        )

    produced_files = _collect_outputs(output_prefix)
    _mirror_outputs_to_sagemaker(produced_files)

    duration = time.time() - start_time
    LOGGER.info("RFdiffusion completed in %.2f seconds", duration)

    response_payload: Dict[str, Any] = {
        "status": "ok",
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_seconds": duration,
        "outputs": _serialise_outputs(produced_files, inline_outputs),
    }

    return response_payload


def _maybe_download_weights(model_directory: str, context: Dict[str, Any], body: Dict[str, Any]) -> None:
    """Download model weights if requested and no checkpoints exist."""

    auto_download = body.get("auto_download_weights") or os.environ.get(
        "RF_AUTO_DOWNLOAD_MODELS", "0"
    ) in {"1", "true", "True"}

    if not auto_download:
        return

    existing = glob.glob(os.path.join(model_directory, "*.pt"))
    if existing:
        LOGGER.info("Model weights already present in %s", model_directory)
        return

    download_script = context.get("download_script")
    if not download_script or not os.path.exists(download_script):
        LOGGER.warning("Download script not available; skipping weight download")
        return

    LOGGER.info("Downloading RFdiffusion weights into %s", model_directory)
    result = subprocess.run(
        ["bash", download_script, model_directory],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        LOGGER.error(
            "Downloading weights failed: rc=%s stderr=%s", result.returncode, result.stderr
        )
        raise RuntimeError(
            "Failed to download RFdiffusion weights",
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
        )


def _collect_outputs(output_prefix: str) -> List[str]:
    """Collect artefacts produced by RFdiffusion for the given prefix."""

    directory = os.path.dirname(output_prefix)
    prefix_basename = os.path.basename(output_prefix)
    pattern = os.path.join(directory, f"{prefix_basename}*")
    files = sorted(glob.glob(pattern))

    LOGGER.debug("Collected %d artefacts from %s", len(files), pattern)
    return files


def _mirror_outputs_to_sagemaker(produced_files: List[str]) -> None:
    """Copy output artefacts into the SageMaker output directory if present."""

    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR")
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)
    for file_path in produced_files:
        try:
            shutil.copy(file_path, output_dir)
        except OSError as exc:  # pragma: no cover
            LOGGER.warning("Unable to copy %s to %s: %s", file_path, output_dir, exc)


def _serialise_outputs(files: List[str], inline_outputs: bool) -> List[Dict[str, Any]]:
    """Serialise produced artefacts for the HTTP response."""

    serialised: List[Dict[str, Any]] = []
    for file_path in files:
        entry: Dict[str, Any] = {
            "path": file_path,
            "filename": os.path.basename(file_path),
        }
        if inline_outputs:
            with open(file_path, "rb") as handle:
                entry["base64"] = base64.b64encode(handle.read()).decode("utf-8")
        serialised.append(entry)

    return serialised