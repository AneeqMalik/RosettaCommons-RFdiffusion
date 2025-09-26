#!/usr/bin/env python
import os, sys, json, subprocess, time, traceback, shlex, glob, pathlib

def log(msg):
    print(f"[transform] {msg}", flush=True)

def run_cmd(cmd, log_file_path, timeout=None):
    log(f"Executing: {cmd}")
    with open(log_file_path, "a", buffering=1) as lf:
        lf.write(cmd + "\n")
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        start = time.time()
        lines = []
        while True:
            line = proc.stdout.readline()
            if line == '' and proc.poll() is not None:
                break
            if line:
                lf.write(line)
                if len(lines) < 10:  # keep some lines for manifest snippet
                    lines.append(line.strip()[:500])
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                raise TimeoutError(f"Command timed out after {timeout}s")
        rc = proc.wait()
    return rc, lines

def maybe_download_models():
    s3_uri = os.environ.get("RFD_MODEL_S3_URI", "").strip()
    model_dir = os.environ.get("RFD_MODEL_DIR", "/app/RFdiffusion/models")
    if not s3_uri:
        log("No RFD_MODEL_S3_URI set; assuming models already present.")
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    marker = os.path.join(model_dir, ".download_complete")
    if os.path.exists(marker):
        log("Model download marker exists; skipping re-download.")
        return
    log(f"Syncing model weights from {s3_uri} to {model_dir}")
    sync_cmd = f"aws s3 sync {shlex.quote(s3_uri)} {shlex.quote(model_dir)}"
    rc = subprocess.call(sync_cmd, shell=True)
    if rc != 0:
        log("ERROR: model download failed")
        sys.exit(1)
    open(marker, "w").close()
    log("Model download complete.")

def main():
    channel = os.environ.get("RFD_INPUT_CHANNEL", "pdbs")
    input_dir = f"/opt/ml/input/data/{channel}"
    output_root = "/opt/ml/output"
    model_dir = os.environ.get("RFD_MODEL_DIR", "/models")
    contig_spec = os.environ.get("RFD_CONTIG_SPEC")
    num_designs = os.environ.get("RFD_NUM_DESIGNS", "3")
    extra_args = os.environ.get("RFD_EXTRA_ARGS", "")
    output_prefix_base = os.environ.get("RFD_OUTPUT_PREFIX", "design")
    timeout = int(os.environ.get("RFD_TIMEOUT_SECONDS", "0")) or None

    if not os.path.isdir(input_dir):
        log(f"Input channel directory not found: {input_dir}")
        sys.exit(1)

    maybe_download_models()

    pdb_files = sorted(
        [p for p in glob.glob(os.path.join(input_dir, "*")) if p.lower().endswith((".pdb",".cif",".ent",".mmcif"))]
    )
    if not pdb_files:
        log("No input PDB/CIF files found; exiting.")
        open(os.path.join(output_root, "manifest.json"), "w").write(json.dumps({"processed":[], "failed":[], "note":"no input files"}))
        return

    processed = []
    failed = []

    for pdb_path in pdb_files:
        stem = pathlib.Path(pdb_path).stem
        out_dir = os.path.join(output_root, stem)
        os.makedirs(out_dir, exist_ok=True)
        log_file = os.path.join(out_dir, "run.log")
        out_prefix = os.path.join(out_dir, output_prefix_base)
        # If the specified model_dir does not exist, fall back to upstream default
        effective_model_dir = model_dir
        if not os.path.isdir(effective_model_dir):
            fallback = "/app/RFdiffusion/models"
            if os.path.isdir(fallback):
                effective_model_dir = fallback
        cmd_parts = [
            "rfdiffusion",
            f"inference.output_prefix={shlex.quote(out_prefix)}",
            f"inference.model_directory_path={shlex.quote(effective_model_dir)}",
            f"inference.input_pdb={shlex.quote(pdb_path)}",
            f"inference.num_designs={shlex.quote(str(num_designs))}",
        ]
        if contig_spec:
            cmd_parts.append(f"contigmap.contigs={shlex.quote(contig_spec)}")
        if extra_args:
            cmd_parts.append(extra_args)

        cmd = " ".join(cmd_parts)
        start_time = time.time()
        try:
            rc, snippet = run_cmd(cmd, log_file, timeout=timeout)
            duration = time.time() - start_time
            if rc != 0:
                raise RuntimeError(f"Return code {rc}")
            produced = sorted([os.path.basename(f) for f in glob.glob(os.path.join(out_dir, "*.pdb"))])
            summary = {
                "input_file": os.path.basename(pdb_path),
                "produced_files": produced,
                "command": cmd,
                "duration_seconds": round(duration,2),
                "log_snippet": snippet,
                "num_designs_env": num_designs,
                "contig_spec": contig_spec,
            }
            with open(os.path.join(out_dir, "summary.json"), "w") as sf:
                json.dump(summary, sf, indent=2)
            processed.append(summary)
            log(f"Completed {pdb_path} in {duration:.1f}s, produced {len(produced)} files.")
        except Exception as e:
            tb = traceback.format_exc().splitlines()[-10:]
            err_info = {
                "input_file": os.path.basename(pdb_path),
                "error": str(e),
                "traceback_tail": tb
            }
            failed.append(err_info)
            with open(os.path.join(out_dir, "error.json"), "w") as ef:
                json.dump(err_info, ef, indent=2)
            log(f"FAILED {pdb_path}: {e}")

    manifest = {
        "processed": processed,
        "failed": failed,
        "totals": {
            "processed": len(processed),
            "failed": len(failed),
            "overall": len(processed)+len(failed)
        }
    }
    with open(os.path.join(output_root, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    if failed:
        log(f"Completed with failures: {len(failed)} failed out of {len(processed)+len(failed)}")
        return
    log("All files processed successfully.")

if __name__ == "__main__":
    main()
