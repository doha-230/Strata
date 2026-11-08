"""Build a Windows, offline IQ3_XXS/256K bundle from pinned original Strata v0.2.12 files.

Run on an internet-connected build PC with Git and Python/pip. The destination PC needs only
the NVIDIA driver and the two user-supplied GSQ-RCO IQ3_XXS model GGUF shards.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
CACHE = DIST / "downloads"
NAME = "strata-win-original-v0.2.12-iq3_xxs-256k-dual"
STAGE = DIST / NAME
SOURCE = DIST / "upstream-v0.2.12-source"
UPSTREAM_URL = "https://github.com/Niko1221/Strata.git"
UPSTREAM_TAG = "v0.2.12"
UPSTREAM_COMMIT = "146df3c468739147f520ac3816de806f71232571"
ENGINE_NAME = "strata-engine-0.1.58-windows-x86_64-cuda-cuda13.tar.gz"
ENGINE_SHA = "3311443ba9172154adc99c48668c4187d60864e3b6cafe5d9aa39b035e4f9f34"
ENGINE_URL = f"https://github.com/Niko1221/Strata/releases/download/{UPSTREAM_TAG}/{ENGINE_NAME}"
PYTHON_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
PYTHON_SHA = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"
MMPROJ_URL = ("https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/"
              "resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/"
              "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
MMPROJ_SHA = "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"
DFLASH_REPO = "https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2"
DFLASH_REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
DFLASH_CONFIG_SHA = "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"
DFLASH_WEIGHTS_SHA = "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"
DFLASH_PACK_NAME = "qwen3.8-flash-next-dflash2.q4_0.pack"
DFLASH_PACK_BYTES = 522458112
DFLASH_PACK_SHA = "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1"
MTP_FILES = {
    "experts.bin": "09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f",
    "dense.bin": "c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a",
    "dense.txt": "991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd",
}
# The original v0.2.12 requirements.txt pins these Windows CPython 3.12 wheels and dependencies.
WINDOWS_WHEELS = (
    "numpy==2.5.3", "jinja2==3.1.6", "regex==2026.9.10", "pyyaml==6.0.3",
    "tqdm==4.70.1", "requests==2.34.2", "pillow==13.0.0", "psutil==7.2.2",
    "markupsafe==3.0.3", "certifi==2026.10.19", "truststore==0.10.5",
    "charset-normalizer==3.5.1", "idna==3.20", "urllib3==2.8.0", "colorama==0.4.6",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path, expected: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        print(f"Downloading {path.name} ...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "strata-offline-builder"})
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(request, timeout=120) as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        part.replace(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path.name}: SHA-256 mismatch ({actual}; expected {expected})")
    return path


def pinned_source() -> None:
    if SOURCE.is_dir() and (SOURCE / "serve" / "server.py").is_file():
        # Re-export below anyway: an old source cache must never silently define the package.
        shutil.rmtree(SOURCE)
    try:
        subprocess.run(["git", "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"],
                       cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        subprocess.run(["git", "fetch", "--no-tags", UPSTREAM_URL, f"refs/tags/{UPSTREAM_TAG}"],
                       cwd=ROOT, check=True)
    commit = subprocess.check_output(["git", "rev-parse", f"{UPSTREAM_COMMIT}^{{commit}}"],
                                     cwd=ROOT, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError("the original Strata source commit changed")
    SOURCE.mkdir(parents=True)
    archive = CACHE / f"strata-source-{UPSTREAM_TAG}.tar"
    with archive.open("wb") as dst:
        subprocess.run(["git", "archive", "--format=tar", UPSTREAM_COMMIT,
                        "serve", "strata_control", "tools", "data", "LICENSE", "requirements.txt"],
                       cwd=ROOT, stdout=dst, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(SOURCE, filter="data")


def extract_engine(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or "/" in member.name or "\\" in member.name:
                raise RuntimeError(f"unexpected engine archive entry: {member.name}")
            with tar.extractfile(member) as src, (dest / member.name).open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    build = json.loads((dest / "BUILD.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((dest / "engine-component.json").read_text(encoding="utf-8-sig"))
    if build.get("version") != "0.1.58" or 86 not in build.get("archs", []):
        raise RuntimeError("the original engine is not the expected sm_86-capable v0.1.58 build")
    if manifest.get("version") != "0.1.58" or manifest.get("backend") != "cuda":
        raise RuntimeError("the original engine component manifest differs")
    for item in manifest["files"]:
        path = dest / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"engine component file differs: {item['path']}")


def install_python(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    (dest / "python312._pth").write_text("python312.zip\n.\nLib\\site-packages\nimport site\n", encoding="utf-8")
    site = dest / "Lib" / "site-packages"
    site.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.12",
                    "--implementation", "cp", "--abi", "cp312", "--target", str(site),
                    "--no-compile", "--no-deps", *WINDOWS_WHEELS], check=True)
    shutil.rmtree(site / "bin", ignore_errors=True)  # build-host entrypoint scripts are not Windows programs


def build() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    engine = fetch(ENGINE_URL, CACHE / ENGINE_NAME, ENGINE_SHA)
    python = fetch(PYTHON_URL, CACHE / "python-embed.zip", PYTHON_SHA)
    mmproj = fetch(MMPROJ_URL, CACHE / "mmproj.gguf", MMPROJ_SHA)
    pinned_source()
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    for folder in ("serve", "strata_control", "tools", "data"):
        shutil.copytree(SOURCE / folder, STAGE / folder)
    shutil.copy2(SOURCE / "LICENSE", STAGE / "STRATA-LICENSE.txt")
    shutil.copy2(SOURCE / "requirements.txt", STAGE / "UPSTREAM-REQUIREMENTS.txt")
    extract_engine(engine, STAGE / "engine")
    install_python(python, STAGE / "python")
    models = STAGE / "models"
    models.mkdir()
    shutil.copy2(mmproj, models / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
    mtp = STAGE / "mtp" / "rt"
    mtp.mkdir(parents=True)
    old_mtp = ROOT / "offline" / "strata-win-iq3_xxs-vision" / "mtp" / "rt"
    for name, digest in MTP_FILES.items():
        src = old_mtp / name
        if sha256(src) != digest:
            raise RuntimeError(f"MTP runtime differs: {name}")
        shutil.copy2(src, mtp / name)
    shutil.copy2(SOURCE / "data" / "draft_vocab.bin", mtp / "draft_vocab.bin")
    packaged = ROOT / "packaging" / "offline_iq3xxs_0212"
    shutil.copy2(packaged / "prepare.py", STAGE / "prepare.py")
    shutil.copy2(packaged / "MANUAL.html", STAGE / "MANUAL.html")
    for name in ("RUN.bat", "RUN-MTP.bat", "RUN-DFLASH2.bat"):
        run = (packaged / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        (STAGE / name).write_bytes(run)
    if any(p.name.endswith("-of-00002.gguf") for p in STAGE.rglob("*.gguf")):
        raise RuntimeError("the user-supplied main model shards must not be bundled")
    provenance = {"source": UPSTREAM_URL, "tag": UPSTREAM_TAG, "commit": UPSTREAM_COMMIT,
                  "engine_release": ENGINE_URL, "engine_sha256": ENGINE_SHA,
                  "python_embed_sha256": PYTHON_SHA, "mmproj_sha256": MMPROJ_SHA,
                  "model": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS", "context_tokens": 262144,
                  "main_model_gguf_included": False,
                  "dflash2_pack_in_separate_release_zip": True,
                  "mtp_runtime": "v0.2.12 tools/mtp_rt.py reproduced the included v0.1.45 runtime files byte-for-byte"}
    (STAGE / "SOURCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    important = [STAGE / "engine" / name for name in
                 ("strata.exe", "strata-vision.exe", "strata_kq.dll", "cublas64_13.dll", "cublasLt64_13.dll",
                  "BUILD.json", "engine-component.json")]
    important += [STAGE / name for name in ("prepare.py", "RUN.bat", "RUN-MTP.bat", "RUN-DFLASH2.bat", "MANUAL.html", "SOURCE.json",
                                                 "serve/server.py", "tools/iq_pack.py", "strata_control/drafting.py",
                                                 "mtp/rt/experts.bin", "mtp/rt/dense.bin", "mtp/rt/dense.txt",
                                                 "mtp/rt/draft_vocab.bin", "models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                                                 "python/python.exe")]
    (STAGE / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(STAGE).as_posix()}\n" for path in important), encoding="utf-8")
    out = DIST / f"{NAME}.zip"
    print(f"Writing {out} ...", flush=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                z.write(path, Path(NAME) / path.relative_to(STAGE))
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failed: {bad}")
    print(f"Ready: {out} ({out.stat().st_size:,} bytes; SHA-256 {sha256(out)})", flush=True)
    return out


def build_dflash2() -> Path:
    """A second ZIP merges into the base folder and stays below GitHub's per-asset limit."""
    base = f"{DFLASH_REPO}/resolve/{DFLASH_REVISION}"
    config = fetch(f"{base}/config.json", CACHE / "dflash2-config.json", DFLASH_CONFIG_SHA)
    weights = fetch(f"{base}/model.safetensors", CACHE / "dflash2-model.safetensors", DFLASH_WEIGHTS_SHA)
    pack = CACHE / DFLASH_PACK_NAME
    if not pack.is_file() or pack.stat().st_size != DFLASH_PACK_BYTES or sha256(pack) != DFLASH_PACK_SHA:
        print("Converting the pinned DFlash2 weights with original v0.2.12 tools/dflash2_pack.py ...", flush=True)
        subprocess.run([sys.executable, str(SOURCE / "tools" / "dflash2_pack.py"),
                        str(weights), str(config), str(pack), "--q4_0"], check=True)
    if pack.stat().st_size != DFLASH_PACK_BYTES or sha256(pack) != DFLASH_PACK_SHA:
        raise RuntimeError("DFlash2 pack does not match the original v0.2.12 pinned result")
    addon = DIST / f"{NAME}-dflash2.zip"
    attribution = {"source": DFLASH_REPO, "revision": DFLASH_REVISION,
                   "source_safetensors_sha256": DFLASH_WEIGHTS_SHA,
                   "conversion": "original Strata v0.2.12 tools/dflash2_pack.py --q4_0",
                   "pack_name": DFLASH_PACK_NAME, "pack_bytes": DFLASH_PACK_BYTES,
                   "pack_sha256": DFLASH_PACK_SHA, "license": "Apache-2.0"}
    print(f"Writing {addon} ...", flush=True)
    with zipfile.ZipFile(addon, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        z.write(pack, Path(NAME) / "dflash2" / DFLASH_PACK_NAME)
        z.write(SOURCE / "data" / "dflash2-tuned" / "LICENSE", Path(NAME) / "dflash2" / "LICENSE")
        z.writestr(str(Path(NAME) / "dflash2" / "SOURCE.json"), json.dumps(attribution, indent=2) + "\n")
    with zipfile.ZipFile(addon) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"DFlash2 ZIP CRC failed: {bad}")
    print(f"Ready: {addon} ({addon.stat().st_size:,} bytes; SHA-256 {sha256(addon)})", flush=True)
    return addon


if __name__ == "__main__":
    build()
    build_dflash2()
