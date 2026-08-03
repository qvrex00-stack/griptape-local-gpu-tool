"""
Griptape Local GPU Tool - Install Script
tool_server (port 8089): Depth Extraction, Pose Extraction, Pose Smoothing, Video Blending

Usage:
    python install.py --griptape-dir "C:/Foundry/Griptape"
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, check=True):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"  ERROR: failed (code {result.returncode})")
        sys.exit(1)
    return result


def find_uv():
    uv = shutil.which("uv")
    if uv: return uv
    for c in [
        Path.home() / "AppData" / "Local" / "uv" / "bin" / "uv.exe",
        Path.home() / ".cargo" / "bin" / "uv.exe",
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ]:
        if c.exists(): return str(c)
    return None


def install_uv():
    print("  Installing uv...")
    if platform.system() == "Windows":
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command",
                        "irm https://astral.sh/uv/install.ps1 | iex"])
    else:
        subprocess.run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    os.environ["PATH"] = (
        str(Path.home() / "AppData" / "Local" / "uv" / "bin") + os.pathsep +
        str(Path.home() / ".cargo" / "bin") + os.pathsep +
        os.environ.get("PATH", "")
    )
    uv = find_uv()
    if not uv:
        print("  ERROR: Restart terminal and retry.")
        sys.exit(1)
    return uv


def detect_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            print(f"  GPU: {line}")
            cc = int(line.split(",")[1].strip().replace(".", ""))
            if cc >= 120: return "nightly_cu128"
            elif cc >= 80: return "stable_cu121"
            else:          return "stable_cu118"
    except Exception:
        pass
    print("  No NVIDIA GPU → CPU mode")
    return "cpu"


def get_torch_cmd(uv, python, gpu_type):
    base = [uv, "pip", "install", "--python", python]
    if gpu_type == "nightly_cu128":
        return base + ["--pre", "--upgrade", "--force-reinstall",
                        "torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/nightly/cu128"]
    elif gpu_type == "stable_cu121":
        return base + ["torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/cu121"]
    elif gpu_type == "stable_cu118":
        return base + ["torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/cu118"]
    return base + ["torch", "torchvision"]


def main():
    parser = argparse.ArgumentParser(description="Install Griptape Local GPU Tool")
    parser.add_argument("--griptape-dir", default=os.environ.get("GRIPTAPE_DIR", ""))
    args = parser.parse_args()

    if not args.griptape_dir:
        print('ERROR: --griptape-dir required')
        sys.exit(1)

    griptape_dir = Path(args.griptape_dir).resolve()
    repo_dir     = Path(__file__).parent.resolve()

    print(f"\nGriptape dir : {griptape_dir}")
    print(f"Repo dir     : {repo_dir}")

    if not griptape_dir.exists():
        print(f"ERROR: {griptape_dir} not found"); sys.exit(1)

    # [0] uv
    print("\n[0/4] Checking uv...")
    uv = find_uv() or install_uv()

    # [1] GPU
    print("\n[1/4] Detecting GPU...")
    gpu_type = detect_gpu()

    # [2] tool_server 복사
    print("\n[2/4] Setting up tool_server...")
    server_dst = griptape_dir / "tool_server"
    server_src = repo_dir / "tool_server"

    if not server_dst.exists():
        shutil.copytree(server_src, server_dst, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
        print(f"  Created: {server_dst}")
    else:
        for f in ["server.py", "requirements.txt"]:
            shutil.copy2(server_src / f, server_dst / f)
        print(f"  Updated: {server_dst}")

    # [3] .venv
    print("\n[3/4] Setting up tool_server venv...")
    venv_dir = server_dst / ".venv"
    python   = str(venv_dir / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python"))

    if not venv_dir.exists():
        run([uv, "venv", str(venv_dir), "--python", "3.12"])

    print(f"  Installing PyTorch ({gpu_type})...")
    run(get_torch_cmd(uv, python, gpu_type))
    print("  Installing requirements...")
    run([uv, "pip", "install", "--python", python,
         "-r", str(server_dst / "requirements.txt")])

    # [4] 노드 파일 + JSON
    print("\n[4/4] Installing node files...")
    lib_dst   = griptape_dir / "libraries" / "griptape-nodes-library-tools"
    nodes_dst = lib_dst / "nodes"
    nodes_dst.mkdir(parents=True, exist_ok=True)

    for f in ["depth_extractor.py", "pose_extractor.py", "pose_smoothing.py", "blend_videos.py"]:
        src = repo_dir / "nodes" / f
        if src.exists():
            shutil.copy2(src, nodes_dst / f)
            print(f"  Copied: {f}")
        else:
            print(f"  WARNING: {f} not found")

    shutil.copy2(repo_dir / "griptape-nodes-library-tools.json",
                 lib_dst / "griptape_nodes_library.json")
    print("  Copied: griptape_nodes_library.json")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print(f"\nLibrary : {lib_dst}")
    print(f"Server  : http://127.0.0.1:8089")
    print("\nNext steps:")
    print("  1. Start Griptape Nodes")
    print("  2. Settings > Libraries > Add Library:")
    print(f"       {lib_dst}")
    print("  3. Restart Griptape → tool server auto-starts")


if __name__ == "__main__":
    main()
