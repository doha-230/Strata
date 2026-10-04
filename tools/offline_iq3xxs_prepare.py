"""Prepare the two user-supplied IQ3_XXS GGUF shards without network access."""
from __future__ import annotations

import argparse
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


def check_files() -> tuple[GGUFFile, GGUFFile]:
    for path in [ROOT / "engine" / "strata.exe", ROOT / "engine" / "strata-vision.exe",
                 ROOT / "data" / "expert-profile.bin", MTP / "experts.bin", MTP / "dense.bin",
                 MTP / "dense.txt", MTP / "draft_vocab.bin", MMPROJ, *SHARDS]:
        if not path.is_file():
            raise RuntimeError(f"필요한 파일이 없습니다: {path}")
    if MMPROJ.stat().st_size < 500_000_000:
        raise RuntimeError(f"이미지 인코더 파일이 너무 작습니다: {MMPROJ}")
    for path in SHARDS:
        if path.stat().st_size < 1_000_000_000:
            raise RuntimeError(f"GGUF 파일이 너무 작습니다: {path}")
    a, b = (GGUFFile(path) for path in SHARDS)
    for path, gguf in zip(SHARDS, (a, b)):
        expected = gguf.data_start + max((t.offset + (t.expected_bytes() or 0) for t in gguf.tensors), default=0)
        if path.stat().st_size < expected:
            raise RuntimeError(f"GGUF가 잘렸습니다: {path.name} ({path.stat().st_size} / {expected} bytes)")
    if not any(t.type_name == "IQ3_XXS" for t in a.tensors):
        raise RuntimeError("첫 번째 GGUF는 GSQ-RCO IQ3_XXS 양자화 파일이 아닙니다.")
    if not any(t.name == "per_layer_token_embd.weight" for t in b.tensors):
        raise RuntimeError("두 번째 GGUF에 PLE 테이블이 없습니다. 원본 GSQ-RCO 두 조각을 확인하세요.")
    return a, b


def check_driver() -> None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA 드라이버 또는 nvidia-smi를 찾지 못했습니다.") from exc
    print(result.stdout.strip(), flush=True)
    if "A6000" not in result.stdout.upper():
        print("주의: A6000이 감지되지 않았습니다. GPU가 다른 경우 호환성을 확인하세요.", flush=True)
    try:
        version = int(result.stdout.strip().split(",")[1].strip().split(".")[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("NVIDIA 드라이버 버전을 확인하지 못했습니다.") from exc
    if version < 580:
        raise RuntimeError(f"NVIDIA 드라이버 {version}은 너무 오래되었습니다. 580 이상이 필요합니다.")


def prepare() -> None:
    check_driver()
    check_files()
    if not (PACK / "native_experts.txt").exists() or not (PACK / "tokenizer" / "vocab.json").exists():
        print("GGUF에서 Strata pack과 토크나이저를 준비합니다 ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"),
                        "--gguf", str(SHARDS[0]), "--out", str(PACK)], check=True, cwd=ROOT)
    cuda_dir = ROOT / "python" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
    if not (cuda_dir / "cublas64_13.dll").exists() or not (cuda_dir / "cudart64_13.dll").exists():
        raise RuntimeError("패키지의 CUDA DLL이 누락되었습니다.")
    cfg = {
        "exe": str(ROOT / "engine" / "strata.exe"),
        "args": ["--pack", str(PACK), "--native", str(SHARDS[0]),
                 "--ple-gguf", str(SHARDS[1]), "--expert-profile", str(ROOT / "data" / "expert-profile.bin"),
                 "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
                 "--mtp", str(MTP), "--max-context", "32768", "--kv", "int8",
                 "--vision", "--vram-reserve-mib", "700"],
        "cwd": str(ROOT), "tokenizer": str(PACK / "tokenizer"),
        "model_name": "qwen3.8-flash-next-iq3_xxs", "log": str(ROOT / "strata-iq3_xxs.log"),
        "lib_dirs": [str(cuda_dir)], "gpu": 0, "port": 8080, "host": "127.0.0.1",
        "vision": {"exe": str(ROOT / "engine" / "strata-vision.exe"), "mmproj": str(MMPROJ),
                   "model": str(SHARDS[0]), "gpu": True, "max_tokens": 1024},
    }
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("오프라인 준비 완료. 이미지 입력과 MTP가 활성화되었습니다.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="드라이버와 필수 파일만 확인")
    args = parser.parse_args()
    try:
        if args.check:
            check_driver()
            check_files()
            print("드라이버와 파일 확인 완료.")
        else:
            prepare()
        return 0
    except Exception as exc:
        print(f"준비 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
