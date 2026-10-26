# Strata: 폐쇄망 Windows 배포판

이 저장소는 [원본 Strata v0.1.45](https://github.com/Niko1221/Strata/releases/tag/v0.1.45)를 기반으로 합니다. Windows PC에서 **Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS**를 인터넷 연결 없이 실행할 수 있도록 패키지를 제공합니다. GPU 이미지 입력과 MTP를 기본으로 켰습니다.

실행에 필요한 Windows 엔진, Python, CUDA 런타임, 이미지 인코더와 `mmproj`, MTP 가중치는 포함했습니다. **메인 LLM GGUF 두 조각과 NVIDIA 시스템 드라이버는 포함하지 않았습니다.** 패키지는 압축 무결성을 확인했지만, Windows 실기기 실행은 아직 검증하지 못했습니다.

## 받기

| 파일 | 용도 |
| --- | --- |
| [Release ZIP](https://github.com/doha-230/Strata/releases/download/v0.1.45-offline-iq3xxs-vision/strata-win-iq3_xxs-vision.zip) | 폐쇄망 PC로 옮겨 압축 해제하는 완성 패키지 |
| [A6000 sm86 PF 기본 ON 빌드](https://github.com/doha-230/Strata/releases/tag/v0.1.45-sm86-v2-default-on) | 기존 패키지의 엔진만 교체하는 선택형 빌드. A6000 실기기 성능은 미측정 |
| [압축 전 파일](offline/strata-win-iq3_xxs-vision/) | Git에서 개별 파일을 관리하는 폴더. 큰 파일은 이 저장소의 Git LFS 사용 |
| [한국어 HTML 설명서](docs/OFFLINE_WINDOWS_IQ3XXS_KO.html) | 파일 위치, 설정 변경, 메모리 사용 및 문제 해결 |

ZIP SHA-256: `886e7bb1a8df50ac95051ab7f29b4bb66a5b7c312252fe36fcd0d006c25376b2` (2,039,005,358바이트). 변경 사항은 [v0.1.45 릴리즈 노트](docs/releases/v0.1.45-offline-iq3xxs-vision.md)에 있습니다.

폐쇄망 PC에는 **Release ZIP**을 가져가세요. GitHub의 저장소 소스 ZIP은 Git LFS 파일을 받는 방법으로 사용하지 마세요.

## 실행 조건과 추가 파일

| 항목 | 조건 |
| --- | --- |
| 시스템 | Windows 10/11 x64, AVX2 지원 CPU |
| 드라이버 | NVIDIA 580 이상. PC에 별도 설치한 뒤 재시작 |
| 저장 공간 | 모델 복사와 첫 준비를 위해 NTFS SSD에 약 85GB 이상 여유 공간 권장 |
| 가상 메모리 | Windows 페이지 파일을 **시스템 관리**로 설정 권장 |

원본 [ISTA-DASLab 모델 페이지](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS)의 두 파일(합계 약 75.8GB)을 인터넷이 되는 PC에서 받은 뒤 폐쇄망 PC로 옮겨야 합니다. 다른 양자화나 Coder·Swift·Unsloth 모델의 GGUF와 섞지 마세요.

## 실행

1. Release ZIP을 NTFS SSD에 풀고, 필요하면 NVIDIA 드라이버를 설치한 뒤 재시작합니다.
2. 아래 **두 파일만** 압축 해제한 폴더의 `models`에 원래 이름 그대로 복사합니다. `mmproj`는 이미 들어 있습니다.

   ```text
   strata-win-iq3_xxs-vision\models\
   ├─ Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
   ├─ Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
   └─ mmproj-Qwen3.8-Flash-Next-BF16.gguf
   ```

3. `RUN.bat`을 실행합니다. 첫 실행에서는 GGUF를 검사하고 Strata pack과 토크나이저를 만듭니다. 완료 후 `http://127.0.0.1:8080/health`에서 `"loaded": true`와 `"images": true`를 확인하고, `http://127.0.0.1:8080`에서 사용합니다.

현재 소스 또는 장문 오버레이를 적용한 패키지의 기본 설정은 256K 컨텍스트, int8 KV 스트리밍(`--kv-resident 32768`), GPU 이미지 입력, MTP `--spec 4`, GPU 0번, 로컬 주소 `127.0.0.1:8080`입니다. 기존 v0.1.45 릴리스 ZIP 자체는 32K입니다. 다른 PC에서 서버에 접속하도록 변경할 때는 반드시 API 키를 설정하세요. 설정은 `prepare.py`에서 고칩니다. `RUN.bat`이 시작할 때마다 `strata-iq3_xxs.json`을 다시 만들기 때문입니다. 자세한 값과 변경 방법은 [HTML 설명서](docs/OFFLINE_WINDOWS_IQ3XXS_KO.html)에 있습니다.

현재 소스에서 패키지를 새로 만들면 `RUN-32K.bat`, `RUN-A6000-128K.bat`, `RUN-A6000-256K.bat`도 들어갑니다. 128K·256K 실행은 int8 KV를 RAM으로 스트리밍하고 GPU에는 32K만 두어 전문가 캐시 공간을 확보합니다. A6000 전용 실행 파일은 `STRATA_PIN_BY_NEED=1`도 켭니다. 기존 v0.1.45 릴리스 ZIP에는 이 실행 파일이 없으므로 작은 오버레이를 별도로 적용해야 합니다. A6000 실기기 속도는 아직 측정하지 않았습니다.

Windows에서 Xeon W-2245와 RTX A6000이 함께 확인되면 `prepare.py`가 긴 입력용 `--prefill auto:32768`과 `--host-core last`를 자동 적용합니다. 128K·256K에서는 `--prompt-cache-spread`도 켜서 앱이 예전 메시지를 수정한 뒤 다시 보낼 때 대화 전체를 다시 읽는 일을 줄입니다. CPU 작업자 수는 엔진의 물리 코어 자동 선택을 유지합니다. `STRATA_HOST_CORE`를 직접 지정했다면 그 값을 우선하고, 체크포인트 분산만 끄려면 `STRATA_PROMPT_CACHE_SPREAD=0`, 프로필 전체를 끄려면 `STRATA_W2245_TUNE=0`으로 실행하세요. 이 PC의 속도 개선 폭은 측정되지 않았습니다.

새로 빌드한 sm86 전용 엔진은 공식 Strata의 9개 sm86 생성 설정과 누락됐던 gate/up·SwiGLU 결합 커널도 포함합니다. `STRATA_SM86_PERF=0`으로 생성 설정을 끌 수 있습니다. 기존 v0.1.45 sm86 v2 교체 EXE에는 이 생성 경로가 들어 있지 않으며, RTX A6000에서 생성 속도는 아직 측정되지 않았습니다.

## 자동 캘리브레이션

첫 실행과 pack 생성이 끝난 뒤 서버 창을 닫고 `CALIBRATE.bat`을 실행하세요. 실제 PC에서 생성 속도를 측정하고 `calibration-iq3_xxs.json`에 결과를 저장합니다. 이후 `RUN.bat`이 PCIe 작업 분배, MTP 추측 기준과 CPU 작업 스레드 수를 **자동 적용**합니다. 결과를 보고 `prepare.py`를 수정할 필요가 없습니다. 기본값이 더 빠르면 기본값을 유지하고, 파일을 지우면 기본값으로 돌아갑니다.

GPU·CPU·드라이버나 주요 모델 설정이 바뀌면 저장된 결과를 적용하지 않습니다. 새 환경에서는 `CALIBRATE.bat`으로 다시 측정하세요. 캘리브레이션 중 서버와 다른 GPU 작업은 종료해야 합니다.

## Git으로 받기

Git을 쓰는 PC에는 Git LFS가 필요합니다. 대용량 객체는 이 저장소의 Git LFS에 저장합니다.

```sh
git lfs install
git clone https://github.com/doha-230/Strata.git
cd Strata
git lfs pull
```

파일은 `offline/strata-win-iq3_xxs-vision/`에 있습니다. Git을 사용하는 경우에도 메인 GGUF 두 조각은 직접 넣어야 합니다. 폐쇄망으로 옮길 때는 LFS가 필요 없는 Release ZIP을 사용하면 됩니다.

## 이전 ZIP에서 업데이트

새 ZIP을 새 폴더에 풀고, 기존 폴더의 메인 GGUF 두 조각을 새 폴더의 `models`로 복사한 뒤 `RUN.bat`을 실행합니다. 기존 설정·캘리브레이션 파일은 복사하지 않습니다. 속도 측정값은 새 엔진에서 `CALIBRATE.bat`으로 다시 만듭니다. 자세한 절차는 HTML 설명서에 있습니다.

## 구성과 출처

- [패키지 생성 스크립트](tools/build_offline_windows_iq3xxs.py)는 인터넷이 되는 빌드 PC에서 Windows 엔진 v0.1.45, Python 3.12, CUDA 13 런타임과 필요한 파일을 모읍니다.
- [실행 준비 스크립트](tools/offline_iq3xxs_prepare.py)는 GGUF, 드라이버와 필수 파일을 확인한 뒤 로컬 실행 설정을 만듭니다.
- Strata 본체와 일반 설치 문서는 [원본 프로젝트](https://github.com/Niko1221/Strata)를 바탕으로 합니다. 이 패키지의 실기기 실행 속도는 아직 측정하지 않았습니다.
- Strata 소스는 [MIT 라이선스](LICENSE)를 따릅니다. 모델과 포함된 다른 구성요소에는 각각의 라이선스가 적용됩니다.
