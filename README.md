# Strata: RTX A6000 폐쇄망 Windows 배포판

이 저장소는 [원본 Strata](https://github.com/Niko1221/Strata)의 포크입니다. NVIDIA RTX A6000 48GB VRAM과 시스템 RAM 64GB를 가진 Windows PC에서 **Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS**를 인터넷 연결 없이 실행할 수 있도록 패키지를 제공합니다. GPU 이미지 입력과 MTP를 기본으로 켰습니다.

실행에 필요한 Windows 엔진, Python, CUDA 런타임, 이미지 인코더와 `mmproj`, MTP 가중치는 포함했습니다. **메인 LLM GGUF 두 조각과 NVIDIA 시스템 드라이버는 포함하지 않았습니다.** 패키지는 압축 무결성을 확인했지만, Windows RTX A6000 실기기 실행은 아직 검증하지 못했습니다.

## 받기

| 파일 | 용도 |
| --- | --- |
| [Release ZIP](https://github.com/doha-230/Strata/releases/download/v0.1.38-offline-a6000-iq3xxs-vision/strata-win-a6000-iq3_xxs-vision.zip) | 폐쇄망 PC로 옮겨 압축 해제하는 완성 패키지 |
| [압축 전 파일](offline/strata-win-a6000-iq3_xxs-vision/) | Git에서 개별 파일을 관리하는 폴더. 큰 파일은 Git LFS 사용 |
| [한국어 HTML 설명서](docs/OFFLINE_WINDOWS_IQ3XXS_KO.html) | 파일 위치, 설정 변경, 메모리 사용 및 문제 해결 |

ZIP SHA-256: `4312915fa536a6f881f3a73a63d442951702280a438b1c2310bd5e5af5adc90e` (1,989,032,649바이트).

폐쇄망 PC에는 **Release ZIP**을 가져가세요. GitHub의 저장소 소스 ZIP은 Git LFS 파일을 받는 방법으로 사용하지 마세요.

## PC와 추가 파일

| 항목 | 조건 |
| --- | --- |
| GPU | NVIDIA RTX A6000 48GB. 포함된 본체와 이미지 실행 파일에 `sm_86` 코드가 있음 |
| 메모리 | RAM 64GB. 시작할 때 다른 큰 프로그램을 닫는 것을 권장 |
| 시스템 | Windows 10/11 x64, AVX2 지원 CPU |
| 드라이버 | NVIDIA 580 이상. PC에 별도 설치한 뒤 재시작 |
| 저장 공간 | 모델 복사와 첫 준비를 위해 NTFS SSD에 약 85GB 이상 여유 공간 권장 |
| 가상 메모리 | Windows 페이지 파일을 **시스템 관리**로 설정 권장 |

원본 [ISTA-DASLab 모델 페이지](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS)의 두 파일(합계 약 75.8GB)을 인터넷이 되는 PC에서 받은 뒤 폐쇄망 PC로 옮겨야 합니다. 다른 양자화나 Coder·Swift·Unsloth 모델의 GGUF와 섞지 마세요.

## 실행

1. Release ZIP을 NTFS SSD에 풀고, 필요하면 NVIDIA 드라이버를 설치한 뒤 재시작합니다.
2. 아래 **두 파일만** 압축 해제한 폴더의 `models`에 원래 이름 그대로 복사합니다. `mmproj`는 이미 들어 있습니다.

   ```text
   strata-win-a6000-iq3_xxs-vision\models\
   ├─ Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
   ├─ Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
   └─ mmproj-Qwen3.8-Flash-Next-BF16.gguf
   ```

3. `RUN.bat`을 실행합니다. 첫 실행에서는 GGUF를 검사하고 Strata pack과 토크나이저를 만듭니다. 완료 후 `http://127.0.0.1:8080/health`에서 `"loaded": true`와 `"images": true`를 확인하고, `http://127.0.0.1:8080`에서 사용합니다.

기본 설정은 32K 컨텍스트, int8 KV 캐시, GPU 이미지 입력, MTP `--spec 4`, GPU 0번, 로컬 주소 `127.0.0.1:8080`입니다. 다른 PC에서 서버에 접속하도록 변경할 때는 반드시 API 키를 설정하세요. 설정은 `prepare.py`에서 고칩니다. `RUN.bat`이 시작할 때마다 `strata-iq3_xxs.json`을 다시 만들기 때문입니다. 자세한 값과 변경 방법은 [HTML 설명서](docs/OFFLINE_WINDOWS_IQ3XXS_KO.html)에 있습니다.

## Git으로 받기

Git을 쓰는 PC에는 Git LFS가 필요합니다. 이 저장소가 공개 포크인 관계로 대용량 객체는 [별도 공개 LFS 저장소](https://github.com/doha-230/Strata-A6000-offline-assets)에 저장했습니다. `.lfsconfig`가 그 주소를 지정합니다.

```sh
git lfs install
git clone https://github.com/doha-230/Strata.git
cd Strata
git lfs pull
```

파일은 `offline/strata-win-a6000-iq3_xxs-vision/`에 있습니다. Git을 사용하는 경우에도 메인 GGUF 두 조각은 직접 넣어야 합니다. 폐쇄망으로 옮길 때는 LFS가 필요 없는 Release ZIP을 사용하면 됩니다.

## 구성과 출처

- [패키지 생성 스크립트](tools/build_offline_windows_iq3xxs.py)는 인터넷이 되는 빌드 PC에서 Windows 엔진 v0.1.38, Python 3.12, CUDA 13 런타임과 필요한 파일을 모읍니다.
- [실행 준비 스크립트](tools/offline_iq3xxs_prepare.py)는 GGUF, 드라이버와 필수 파일을 확인한 뒤 로컬 실행 설정을 만듭니다.
- Strata 본체와 일반 설치 문서는 [원본 프로젝트](https://github.com/Niko1221/Strata)를 바탕으로 합니다. 이 포크의 A6000 실행 속도 수치는 아직 측정하지 않았습니다.
- Strata 소스는 [MIT 라이선스](LICENSE)를 따릅니다. 모델과 포함된 다른 구성요소에는 각각의 라이선스가 적용됩니다.
