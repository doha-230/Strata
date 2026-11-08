"""Prepare the pinned upstream Strata 0.2.12 Windows IQ3_XXS bundle without network access."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile  # noqa: E402

BASE = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-"
SHARDS = [ROOT / "models" / f"{BASE}0000{i}-of-00002.gguf" for i in (1, 2)]
MMPROJ = ROOT / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
PACK = ROOT / "packs" / "iq3_xxs"
MTP = ROOT / "mtp" / "rt"
CONFIG = ROOT / "strata-iq3_xxs.json"
DRAFT_PACK = ROOT / "dflash2" / "qwen3.8-flash-next-dflash2.q4_0.pack"
DRAFT_BYTES = 522458112
DRAFT_SHA256 = "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1"
ENGINE_VERSION = "0.1.58"
CONTEXT = 262144
KV_RESIDENT = 32768


def check_driver() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA 드라이버 또는 nvidia-smi를 찾을 수 없습니다.") from exc
    first = result.stdout.strip().splitlines()[0].split(",")
    try:
        driver = tuple(int(x) for x in first[1].strip().split(".")[:2])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("NVIDIA 드라이버 버전을 읽을 수 없습니다.") from exc
    if driver < (580, 65):
        raise RuntimeError(f"드라이버 {first[1].strip()}은 너무 오래되었습니다. Windows CUDA 13 엔진은 580.65 이상이 필요합니다.")
    if "RTX A6000" not in first[0].upper():
        print(f"주의: GPU 0이 RTX A6000이 아닙니다 ({first[0].strip()}). 이 패키지는 A6000용으로 준비했습니다.", flush=True)
    return result.stdout.strip()


def check_dflash_pack() -> None:
    if not DRAFT_PACK.is_file():
        raise RuntimeError("DFlash2 pack이 없습니다. 릴리스의 DFlash2 추가 ZIP을 같은 폴더에 풀어 주세요.")
    if DRAFT_PACK.stat().st_size != DRAFT_BYTES:
        raise RuntimeError("DFlash2 pack 크기가 원본 v0.2.12의 고정값과 다릅니다.")
    digest = hashlib.sha256()
    with DRAFT_PACK.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DRAFT_SHA256:
        raise RuntimeError("DFlash2 pack SHA-256이 원본 v0.2.12의 고정값과 다릅니다.")


def check_files(drafter: str = "dflash2") -> None:
    required = [ROOT / "engine" / "strata.exe", ROOT / "engine" / "strata-vision.exe",
                ROOT / "engine" / "cublas64_13.dll", ROOT / "engine" / "cublasLt64_13.dll",
                ROOT / "data" / "expert-profile.bin", ROOT / "data" / "draft_vocab.bin", MMPROJ, *SHARDS]
    if drafter == "mtp":
        required += [MTP / name for name in ("experts.bin", "dense.bin", "dense.txt", "draft_vocab.bin")]
    for path in required:
        if not path.is_file():
            if drafter == "mtp" and path.parent == MTP:
                raise RuntimeError("MTP 파일이 없습니다. 릴리스의 MTP 추가 ZIP을 같은 폴더에 풀어 주세요.")
            raise RuntimeError(f"필요한 파일이 없습니다: {path}")
    build = json.loads((ROOT / "engine" / "BUILD.json").read_text(encoding="utf-8-sig"))
    if build.get("version") != ENGINE_VERSION or 86 not in build.get("archs", []):
        raise RuntimeError(f"엔진은 CUDA sm_86 빌드 {ENGINE_VERSION}이어야 합니다.")
    if MMPROJ.stat().st_size < 500_000_000:
        raise RuntimeError("이미지 인코더 모델이 너무 작습니다.")
    for path in SHARDS:
        if path.stat().st_size < 1_000_000_000:
            raise RuntimeError(f"GGUF 파일이 너무 작습니다: {path.name}")
    first, second = (GGUFFile(path) for path in SHARDS)
    for path, gguf in zip(SHARDS, (first, second)):
        end = gguf.data_start + max((t.offset + (t.expected_bytes() or 0) for t in gguf.tensors), default=0)
        if path.stat().st_size < end:
            raise RuntimeError(f"GGUF 파일이 잘렸습니다: {path.name}")
    if not any(t.type_name == "IQ3_XXS" for t in first.tensors):
        raise RuntimeError("첫 번째 파일은 지정한 GSQ-RCO IQ3_XXS 양자화 GGUF가 아닙니다.")
    if not any(t.name == "per_layer_token_embd.weight" for t in second.tensors):
        raise RuntimeError("두 번째 GGUF에 PLE 테이블이 없습니다. 지정한 모델의 두 조각을 확인하세요.")
    if drafter == "dflash2":
        check_dflash_pack()


def prepare(drafter: str = "dflash2") -> dict:
    if drafter not in ("mtp", "dflash2"):
        raise ValueError(f"지원하지 않는 생성 보조 방식: {drafter}")
    check_driver()
    check_files(drafter)
    if not (PACK / "native_experts.txt").exists() or not (PACK / "tokenizer" / "vocab.json").exists():
        print("원본 GGUF에서 Strata pack과 토크나이저를 준비합니다. 첫 실행에는 시간이 걸립니다.", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"),
                        "--gguf", str(SHARDS[0]), "--out", str(PACK)], check=True, cwd=ROOT)
    cfg = {
        "exe": str(ROOT / "engine" / "strata.exe"),
        "args": ["--pack", str(PACK), "--native", str(SHARDS[0]), "--ple-gguf", str(SHARDS[1]),
                 "--expert-profile", str(ROOT / "data" / "expert-profile.bin"),
                 "--expert-cache", "auto", "--prefill", "auto", "--mtp", str(MTP), "--spec", "4",
                 "--max-context", str(CONTEXT), "--kv", "int8", "--kv-resident", str(KV_RESIDENT),
                 "--vision", "--vram-reserve-mib", "700"],
        "cwd": str(ROOT), "tokenizer": str(PACK / "tokenizer"),
        "model_name": "qwen3.8-flash-next-iq3_xxs", "arch": "qwen4exp", "backend": "cuda",
        "drafting": drafter, "log": str(ROOT / "strata-iq3_xxs.log"),
        "lib_dirs": [str(ROOT / "engine")], "gpu": 0, "port": 8080, "host": "127.0.0.1",
        "vision": {"exe": str(ROOT / "engine" / "strata-vision.exe"), "mmproj": str(MMPROJ),
                   "model": str(SHARDS[0]), "gpu": True, "max_tokens": 1024},
    }
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"준비 완료: 원본 엔진 {ENGINE_VERSION}, 256K 컨텍스트, int8 KV (GPU 상주 {KV_RESIDENT}토큰), {drafter.upper()}, 이미지 입력.",
          flush=True)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="원본 Strata v0.2.12 폐쇄망 실행 설정")
    parser.add_argument("--drafter", choices=("mtp", "dflash2"), default="dflash2")
    options = parser.parse_args()
    try:
        prepare(options.drafter)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"준비 실패: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
