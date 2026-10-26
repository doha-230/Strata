"""Prepare the two user-supplied IQ3_XXS GGUF shards without network access."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
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
CALIBRATION = ROOT / "calibration-iq3_xxs.json"
TUNED_FLAGS = {"--pcie-frac", "--spec-min-p", "--pool-workers"}
CONTEXT_CHOICES = {32768, 131072, 262144}


def windows_cpu_name() -> str:
    """Read the marketing name; PROCESSOR_IDENTIFIER only contains family/model IDs."""
    if platform.system() != "Windows":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
    except (OSError, ImportError):
        return ""


def w2245_profile(cpu_name: str, gpu_info: str) -> bool:
    """Select the long-prompt profile only for the intended Windows A6000 PC."""
    first_gpu = gpu_info.splitlines()[0].split(",", 1)[0] if gpu_info else ""
    return (os.environ.get("STRATA_W2245_TUNE") != "0"
            and re.search(r"\bXeon\b.*\bW-2245\b", cpu_name, re.IGNORECASE) is not None
            and "RTX A6000" in first_gpu.upper())


def context_args() -> list[str]:
    """Keep long-context KV from displacing the GPU expert cache."""
    raw = os.environ.get("STRATA_CONTEXT_TOKENS", "262144")
    if not raw.isdecimal() or int(raw) not in CONTEXT_CHOICES:
        raise ValueError("STRATA_CONTEXT_TOKENS는 32768, 131072, 262144 중 하나여야 합니다")
    context = int(raw)
    args = ["--max-context", str(context), "--kv", "int8"]
    if context > 32768:
        args += ["--kv-resident", "32768"]
    return args


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


def check_driver() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA 드라이버 또는 nvidia-smi를 찾지 못했습니다.") from exc
    print(result.stdout.strip(), flush=True)
    try:
        version = int(result.stdout.strip().split(",")[1].strip().split(".")[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("NVIDIA 드라이버 버전을 확인하지 못했습니다.") from exc
    if version < 580:
        raise RuntimeError(f"NVIDIA 드라이버 {version}은 너무 오래되었습니다. 580 이상이 필요합니다.")
    return result.stdout.strip()


def calibration_key(cfg: dict, gpu_info: str) -> dict:
    args = cfg["args"]
    values = {}
    for flag in ("--expert-cache", "--prefill", "--spec", "--max-context", "--kv",
                 "--kv-resident", "--vram-reserve-mib"):
        if flag in args:
            values[flag] = args[args.index(flag) + 1]
    values["--vision"] = "--vision" in args
    build = json.loads((ROOT / "engine" / "BUILD.json").read_text(encoding="utf-8"))
    return {
        "schema": 1,
        "gpu_info": gpu_info,
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "engine_version": build["version"],
        "model": cfg["model_name"],
        "gpu": cfg["gpu"],
        "args": values,
        "vision": cfg["vision"],
    }


def validate_settings(settings: object) -> dict[str, str]:
    if not isinstance(settings, dict) or set(settings) - TUNED_FLAGS:
        raise ValueError("알 수 없는 캘리브레이션 설정")
    out = {}
    for flag, value in settings.items():
        if not isinstance(value, str):
            raise ValueError(f"{flag} 값이 문자열이 아닙니다")
        if flag == "--pool-workers":
            if not value.isdecimal() or not 1 <= int(value) <= 256:
                raise ValueError(f"{flag} 값이 범위를 벗어났습니다")
        elif not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{flag} 값이 범위를 벗어났습니다")
        out[flag] = value
    return out


def apply_saved_calibration(cfg: dict, gpu_info: str) -> None:
    if not CALIBRATION.is_file():
        return
    try:
        saved = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        if saved["hardware"] != calibration_key(cfg, gpu_info):
            print("저장된 캘리브레이션이 현재 PC 또는 설정과 달라 기본값을 사용합니다.", flush=True)
            return
        settings = validate_settings(saved["settings"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"캘리브레이션 파일을 읽지 못해 기본값을 사용합니다: {exc}", flush=True)
        return
    import calibrate as CAL
    cfg["args"] = CAL.apply(cfg["args"], settings)
    print("저장된 캘리브레이션 자동 적용: " +
          (", ".join(f"{k} {v}" for k, v in settings.items()) or "기본값이 가장 빠름"), flush=True)


def prepare(*, apply_saved: bool = True) -> tuple[dict, str]:
    selected_context_args = context_args()
    gpu_info = check_driver()
    cpu_name = windows_cpu_name()
    tuned_w2245 = w2245_profile(cpu_name, gpu_info)
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
                 "--expert-cache", "auto", "--prefill", "auto:32768" if tuned_w2245 else "auto",
                 "--spec", "4", "--spec-min-p", "0.5",
                 "--mtp", str(MTP), *selected_context_args, "--vision", "--vram-reserve-mib", "700"],
        "cwd": str(ROOT), "tokenizer": str(PACK / "tokenizer"),
        "model_name": "qwen3.8-flash-next-iq3_xxs", "log": str(ROOT / "strata-iq3_xxs.log"),
        "lib_dirs": [str(cuda_dir)], "gpu": 0, "port": 8080, "host": "127.0.0.1",
        "vision": {"exe": str(ROOT / "engine" / "strata-vision.exe"), "mmproj": str(MMPROJ),
                   "model": str(SHARDS[0]), "gpu": True, "max_tokens": 1024},
    }
    if tuned_w2245 and "STRATA_HOST_CORE" not in os.environ:
        cfg["args"] += ["--host-core", "last"]
    if tuned_w2245 and os.environ.get("STRATA_PROMPT_CACHE_SPREAD") != "0" and "--kv-resident" in selected_context_args:
        cfg["args"].append("--prompt-cache-spread")
    if tuned_w2245:
        print(f"Xeon W-2245 / RTX A6000 프로필: prefill auto:32768, "
              f"host core {'last' if 'STRATA_HOST_CORE' not in os.environ else '환경변수 설정값'}, "
              "CPU 작업자 수 자동 선택, "
              f"프롬프트 체크포인트 분산 {'켜짐' if '--prompt-cache-spread' in cfg['args'] else '꺼짐'}.", flush=True)
    if apply_saved:
        apply_saved_calibration(cfg, gpu_info)
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("오프라인 준비 완료. 이미지 입력과 MTP가 활성화되었습니다. "
          f"컨텍스트 {cfg['args'][cfg['args'].index('--max-context') + 1]} 토큰" +
          (", KV 스트리밍 32768 토큰." if "--kv-resident" in cfg["args"] else "."), flush=True)
    return cfg, gpu_info


def calibrate_pc() -> None:
    cfg, gpu_info = prepare(apply_saved=False)
    import calibrate as CAL
    print("이 PC의 생성 속도를 측정합니다. 진행 중에는 다른 GPU 작업을 종료하세요.", flush=True)
    result = CAL.run(cfg)
    settings = validate_settings(result["settings"])
    saved = {"hardware": calibration_key(cfg, gpu_info),
             "settings": settings, "report": result["report"]}
    tmp = CALIBRATION.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CALIBRATION)
    cfg["args"] = CAL.apply(cfg["args"], settings)
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("캘리브레이션 저장 완료: " + str(CALIBRATION), flush=True)
    print("적용 값: " + (", ".join(f"{k} {v}" for k, v in settings.items())
                     or "기본값이 가장 빠름"), flush=True)
    print("이후 RUN.bat 실행 때 자동 적용됩니다.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="드라이버와 필수 파일만 확인")
    parser.add_argument("--calibrate", action="store_true", help="이 PC를 측정하고 결과를 자동 저장")
    args = parser.parse_args()
    try:
        if args.check:
            check_driver()
            check_files()
            print("드라이버와 파일 확인 완료.")
        elif args.calibrate:
            calibrate_pc()
        else:
            prepare()
        return 0
    except Exception as exc:
        print(f"준비 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
