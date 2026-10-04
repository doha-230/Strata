"""Build the Windows IQ3_XXS vision and MTP bundle from pinned upstream assets.

Run on an internet-connected machine with Python, pip, numpy and PyYAML. The produced zip
needs only an NVIDIA driver and the user's two GGUF shards on the destination PC.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
NAME = "strata-win-a6000-iq3_xxs-vision"
STAGE = DIST / NAME
CACHE = DIST / "downloads"
PYTHON_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
ENGINE_URL = "https://github.com/Niko1221/Strata/releases/download/v0.1.38/strata-windows-x64.zip"
MMPROJ_URL = ("https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/"
              "resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/"
              "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
MMPROJ_SHA256 = "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip"
PACKAGES = [
    "numpy==2.5.3", "jinja2==3.1.6", "regex==2026.9.10", "pyyaml==6.0.3",
    "tqdm==4.70.1", "requests==2.34.2", "pillow==12.3.0", "psutil==7.2.2",
    "markupsafe==3.0.3", "certifi==2026.7.22", "charset-normalizer==3.5.1",
    "idna==3.20", "urllib3==2.8.0", "colorama==0.4.6",
    "nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96",
]


def fetch(url: str, dst: Path) -> None:
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dst.name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "strata-offline-builder"})
    with urllib.request.urlopen(req, timeout=120) as src, dst.with_suffix(dst.suffix + ".part").open("wb") as out:
        shutil.copyfileobj(src, out)
    dst.with_suffix(dst.suffix + ".part").replace(dst)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    py_zip, eng_zip, mmproj = (CACHE / "python-embed.zip", CACHE / "engine.zip", CACHE / "mmproj.gguf")
    fetch(PYTHON_URL, py_zip)
    fetch(ENGINE_URL, eng_zip)
    fetch(MMPROJ_URL, mmproj)
    if hashlib.sha256(mmproj.read_bytes()).hexdigest() != MMPROJ_SHA256:
        raise RuntimeError("mmproj SHA-256 differs from the checked file")

    rt = DIST / "mtp-build" / "rt"
    if not all((rt / x).is_file() for x in ("experts.bin", "dense.bin", "dense.txt")):
        raw = DIST / "mtp-build"
        raw.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(raw)], check=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(raw)], check=True)
        llama_zip = CACHE / "llama.cpp.zip"
        fetch(LLAMA_URL, llama_zip)
        with zipfile.ZipFile(llama_zip) as z:
            gguf_names = [x for x in z.namelist() if "/gguf-py/gguf/" in x]
            z.extractall(DIST / "llama-source", members=gguf_names)
        gguf_py = next((DIST / "llama-source").glob("*/gguf-py"))
        env = dict(os.environ, STRATA_GGUF_PY=str(gguf_py))
        gguf_out = raw / "mtp-q2_0.gguf"
        subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(raw),
                        "--experts", "q2_0", "--out", str(gguf_out)], check=True, env=env)
        subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(gguf_out),
                        "--out", str(rt)], check=True, env=env)
    shutil.copy2(ROOT / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
    shutil.copytree(rt, STAGE / "mtp" / "rt", dirs_exist_ok=True)

    py_dir = STAGE / "python"
    py_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(py_zip) as z:
        z.extractall(py_dir)
    (py_dir / "python312._pth").write_text(
        "python312.zip\n.\nLib\\site-packages\nimport site\n", encoding="utf-8")

    site = py_dir / "Lib" / "site-packages"
    shutil.rmtree(site, ignore_errors=True)
    site.mkdir(parents=True, exist_ok=True)
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
           "--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.12",
           "--implementation", "cp", "--abi", "cp312", "--target", str(site),
           "--no-compile", *PACKAGES]
    subprocess.run(pip, check=True)

    eng_dir = STAGE / "engine"
    eng_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(eng_zip) as z:
        z.extractall(eng_dir)
    meta = json.loads((eng_dir / "BUILD.json").read_text())
    if tuple(map(int, meta["version"].split("."))) < (0, 1, 38):
        raise RuntimeError("engine must be v0.1.38 or newer")
    if not {"strata.exe", "strata-vision.exe"} <= {p.name for p in eng_dir.iterdir()}:
        raise RuntimeError("Windows engine or image encoder is missing")

    models_dir = STAGE / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy2(mmproj, models_dir / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
    for folder in ("serve", "tools"):
        dest = STAGE / folder
        dest.mkdir(exist_ok=True)
        for p in (ROOT / folder).iterdir():
            if p.is_file() and (p.suffix in (".py", ".html", ".css", ".js", ".json", ".svg") or p.name == "favicon.ico"):
                shutil.copy2(p, dest / p.name)
        if folder == "serve":
            for p in (ROOT / folder).iterdir():
                if p.is_dir() and p.name not in ("__pycache__",):
                    shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
    shutil.copytree(ROOT / "data", STAGE / "data", dirs_exist_ok=True)
    shutil.copy2(ROOT / "LICENSE", STAGE / "STRATA-LICENSE.txt")
    shutil.copy2(ROOT / "tools" / "offline_iq3xxs_prepare.py", STAGE / "prepare.py")
    shutil.copy2(ROOT / "tools" / "offline_iq3xxs_run.bat", STAGE / "RUN.bat")
    (STAGE / "RUN.bat").write_bytes((STAGE / "RUN.bat").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    shutil.copy2(ROOT / "docs" / "OFFLINE_WINDOWS_IQ3XXS_KO.html", STAGE / "MANUAL.html")

    if any(p.name.endswith("-of-00002.gguf") for p in STAGE.rglob("*.gguf")):
        raise RuntimeError("the bundle must not contain the main LLM GGUF shards")
    important = [eng_dir / "strata.exe", eng_dir / "strata-vision.exe",
                 models_dir / "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                 STAGE / "mtp" / "rt" / "experts.bin", STAGE / "mtp" / "rt" / "dense.bin",
                 py_dir / "python.exe", site / "nvidia" / "cu13" / "bin" / "x86_64" / "cublas64_13.dll",
                 STAGE / "MANUAL.html"]
    checksums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(STAGE).as_posix()}"
                 for p in important]
    (STAGE / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    out = DIST / f"{NAME}.zip"
    print(f"Writing {out} ...", flush=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DIST))
    print(f"Ready: {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
