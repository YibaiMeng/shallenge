import random
import subprocess


import modal

app = modal.App()

binary_path = "build/shallenge"
mount = modal.Mount.from_local_file(binary_path, remote_path="/workspace/shallenge")
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
    add_python="3.12",
).copy_mount(mount, remote_path="/")


@app.function(gpu="a10g", image=image)
def test_nvidia_smi():

    output = subprocess.check_output(["nvidia-smi"], text=True)
    print(output)


@app.function(gpu="a10g", image=image, timeout=8 * 60 * 60)
def run_shallenge(config: dict):

    index = config["index"]
    seed = config["seed"]
    hashes = config["hashes"]
    grid_size = config["grid_size"]
    block_size = config["block_size"]
    dry_run = config["dry_run"]

    print("Running shallenge instance: ", config)

    popen_args = [
        "/workspace/shallenge",
        "--seed",
        str(seed),
        "--hashes",
        str(hashes),
        "--grid_size",
        str(grid_size),
        "--block_size",
        str(block_size),
    ]
    if dry_run:
        popen_args.append("--dry_run")

    process = subprocess.Popen(
        popen_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for line in iter(process.stderr.readline, b""):
        print(f"[{index}] " + line.decode("utf-8").strip())

    # Ensure the process has finished
    process.stdout.close()
    process.stderr.close()
    process.wait()


@app.local_entrypoint()
def main():

    num_instances = 3
    seed = [random.randint(1000000000, 99999999999) for i in range(num_instances)]
    # Since on A10G we can get 5.60 GH/S, and on A100 we can only get 8.2 GH/s, and A10G are more than 3 times less expensive than A100, we should
    # spin up 3 A10G.
    print(f"Running {num_instances} processes in parallel")
    for _ in run_shallenge.map(
        [
            {
                "index": i,
                "seed": seed[i],
                "hashes": 150,
                "grid_size": 288,
                "block_size": 1024,
                "dry_run": False,
            }
            for i in range(num_instances)
        ]
    ):
        print("Finished")
