import subprocess
import modal
import os

app = modal.App(name="megakernels-ablation")

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12")
    .run_commands("apt update && apt install -y git")
    .uv_pip_install("torch", "nvidia-cutlass-dsl[cu13]", "nvitop")
    .pip_install("uv")
    .workdir("/data")
)

vol = modal.Volume.from_name("megakernel_volume", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, gpu="RTX-PRO-6000", timeout=7200)
def runner(output_dir: str = "benchmarks/results"):
    import subprocess
    import os
    if not os.path.exists(".venv"):
        subprocess.run(["uv", "venv", "--python", "3.12"])
    subprocess.run(["uv", "sync", "--all-extras"])
    result = subprocess.run(
        ["uv", "run", "python3", "-u", "benchmarks/run_ablation_benchmarks.py",
         "--output", f"{output_dir}/results.csv"],
    )
    subprocess.run(
        ["uv", "run", "python3", "-u", "benchmarks/plot_ablation_results.py",
         "--input", f"{output_dir}/results.csv", "--output-dir", output_dir],
    )
    vol.commit()
    return result.returncode


blacklist = ["__pycache__", "ptx", "cubin", "sass", "ncu-rep", "png", "npy", ".venv", ".git", "output"]


@app.local_entrypoint()
def main(output_dir: str = "benchmarks/results"):
    with vol.batch_upload(force=True) as batch:
        for file in os.listdir("."):
            if any(item in file for item in blacklist):
                continue
            if file == "benchmarks":
                for sub in os.listdir(file):
                    if sub == "results":
                        continue
                    path = os.path.join(file, sub)
                    print(f"Uploading file {path}")
                    batch.put_file(path, path)
                continue
            if os.path.isdir(file):
                print(f"Uploading Dir {file}")
                batch.put_directory(file, file)
            else:
                print(f"Uploading file {file}")
                batch.put_file(file, file)
    rc = runner.remote(output_dir)
    print(f"\nRunner exited with code: {rc}")
