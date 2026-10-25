# Basalt 최적화의 A6000 이식 가능성 조사

조사일: 2026-10-25. 대상: Windows x64, RTX A6000 sm86, Xeon W-2223,
RAM 128 GB, Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, 실제 입력 128K~256K, 단일 요청.

## 판단과 조사 범위

**구현 가능성이 높다. 주요 작업은 GPU 커널과 실행 경로 선택, 빌드 조건의 수정이다.**
CPU 엔진이나 서버를 전부 다시 작성할 근거는 발견하지 못했다.
그러나 sm120을 sm86으로 바꾸는 설정 수정만으로는 실행되지 않는다.
기존 일반 CUDA 경로를 연결하면 첫 실행까지의 작업을 줄일 수 있고,
성능을 확보하려면 일부 핵심 경로를 Ampere에 맞춰 최적화해야 한다.

성능 개선 여지는 있지만 **A6000에서 Strata 대비 2배, 또는 특정 개선율을 입증한 결과는 없다.**
호환성 확보와 성능 확보는 별도의 완료 조건이다. 일반 경로로 우회한 첫 버전은
현재 Strata와 비슷하거나 느릴 수도 있다.

조사 기준:

- 현재 Strata: `fa112e511a15c0841aaef1016525b19697f0bd56`.
- Basalt: [`8d73a39477690e09f5fcc037bbe62f0a06481743`](https://github.com/jesdga95/basalt/tree/8d73a39477690e09f5fcc037bbe62f0a06481743).
- Basalt 소스를 별도 임시 디렉터리에 받아 빌드 설정, 커널, 호출부, 대체 경로와 문서를 정적으로 조사했다.
- 호스트에 설치된 CUDA 도구와 GPU는 없지만, 2026-10-26에 임시 CUDA 13.0 도구를 받아 Linux sm86 빌드에서
  `prefill_fused_iq_test`와 `strata` 실행 파일의 컴파일·링크를 확인했다. Windows 빌드, GPU 실행, 정확도와 속도는
  검증하지 않았다.
- 아래 일수와 파일 수는 소스 조사에 근거한 작업량 추정이며 실측 개발 기간이 아니다.
- ninfer3090 비교는 이번 범위에 포함하지 않았다. 우선 Basalt 내부의 대체 경로만으로 이식 경로를 확인했다.

## 실제 수정 지점

| 영역 | 확인한 제약 | 첫 실행을 위한 처리 | 이후 성능 작업 |
| --- | --- | --- | --- |
| 빌드·장치 검사 | CMake와 런타임이 sm120만 허용 | sm86 허용, 기능별 지원 판정 추가 | 기능 선택 로그와 회귀 검사 |
| Dense tensor core | `dense_tc.cu`의 주요 본체는 sm90 이상, 하위 아키텍처는 trap. 지원 함수는 형상·타입 중심 판정 | sm86에서 선택되지 않도록 하고 기존 MMVQ 경로 연결 | Ampere INT8 MMA와 `cp.async` 기반 다중 토큰 커널 |
| GDN recurrent | `dec_gdn_rec.cu`의 cluster 사용, `dec_gdnf_mode()`는 고정값 3 | cluster 코드의 컴파일 분리와 기존 conv/AB/step/norm 경로 연결 | 측정상 병목이면 sm86용 부분 융합 |
| HC/router | `hc_fused.cu`에 TMA 명령과 장치 조건 | HC8 데이터 관리·양자화는 유지하고 `hc8_gr`/`gr_read_group` 경로 사용 | 필요할 때 smaller-block 구성의 융합 구현 |
| 여러 연산을 묶은 mid chain | `mid_chain.cu`가 위 CUDA 파일들을 직접 include | 지원 판정만 끄지 말고 포함된 장치 코드까지 컴파일 분리 | 전체 융합 재구현은 첫 단계에서 제외 |
| PDL | 공용 헤더는 sm86 대체 처리가 있으나 `gu2.cuh`에는 직접 쓴 `griddepcontrol` 명령이 남음 | 직접 명령도 아키텍처별로 보호하고 일반 순차 launch 사용 | 수학 연산과 gate/up 재사용은 유지 |
| Device-planned MoE | `dec_moe.cuh`의 TMA 방식은 sm86 불가 | TMA 인스턴스 분리, 기존 grouped/non-TMA 경로 검증 | CPU miss와 GPU 작업의 겹침, 타일 크기 조정 |
| Prefill MMQ | `mmq_common.cuh`에 sm120 미만 컴파일 오류와 PDL intrinsic 직접 호출 | 컴파일 제한과 PDL 처리 수정, 공유 메모리·타일 설정 검증 | sm86별 타일/occupancy 선택 |
| NVFP4 | 별도 FP4 라이브러리가 `120a-real`로 항상 빌드됨 | IQ3_XXS 전용 구성에서는 호출부·링크까지 조건부 제외하거나 미사용 별도 라이브러리로 유지 | NVFP4 이식은 범위 밖 |

근거 소스:
[CMake](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/CMakeLists.txt),
[device](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/core/device.cu#L151),
[dense_tc](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/dense_tc.cu),
[GDN](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/dec_gdn_rec.cu),
[HC](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/hc_fused.cu),
[mid chain](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/mid_chain.cu),
[gu2](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/gu2.cuh),
[MoE](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/dec_moe.cuh),
[MMQ](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/prefill/mmq/mmq_common.cuh).

TMA는 Hopper sm90 이상 기능이다. Ampere에는 비동기 복사 기능이 있으므로
알고리즘을 유지하면서 전송·동기화 방식을 다시 구성할 수 있다.
sm86의 블록당 공유 메모리 한도도 고려해야 한다.
[NVIDIA TMA 문서](https://nvidia.github.io/cccl/unstable/cccl/tma.html),
[Ampere tuning guide](https://docs.nvidia.com/cuda/archive/12.6.3/ampere-tuning-guide/index.html).

특히 HC는 GPU 세대 이름만 바꾸는 문제가 아니다. 현재 상수에서 `CN_ND=88`,
`CN_MIN_NU=80`이며 지원 조건은 `sms - CN_ND >= CN_MIN_NU`다.
즉 해당 경로는 최소 168 SM을 요구한다. 이 구성 자체를 A6000에 그대로 적용할 수 없다.
다만 이것은 전체 Basalt의 요구 조건이 아니라 특정 융합 경로의 조건이다.

## 재사용 가능한 부분

- **GDN 일반 경로:** `verify.cpp`에 conv/L2, AB, recurrent step/norm을 따로 실행하는 경로가 있다.
- **HC8:** `hc8_gr.cu`의 분리된 norm/down/up 경로가 있다. HC8 데이터가 같은 파일에 있다는 이유로
  `hc_fused.cu` 전체를 삭제하면 안 된다.
- **QSA 선택·샘플링:** cluster 지원 검사와 일반 경로가 이미 있다. 재작성보다 실제 긴 입력에서의 검증이 우선이다.
- **MTP:** conditional graph 캡처 실패 시 per-step graph로 돌아가는 경로가 있다.
  조건부 그래프 전체를 sm86 불가로 간주할 근거는 없다. 해당 Windows 드라이버에서 캡처·실행을 확인해야 한다.
- **IQ 프리필:** `moe_fused_iq.cu`는 sm80 이상 INT8 MMA 및 비동기 복사 경로를 갖고 있다.
  다만 현재 Strata에도 같은 계열의 fused IQ 프리필이 있으므로 이를 신규 이득으로 모두 계산하면 안 된다.
- **Windows:** 플랫폼 코드, MSVC 빌드, CPU 테스트 기반이 이미 있다. 운영체제 이식을 처음부터 할 필요는 없다.

호출 근거:
[verify](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/core/verify.cpp#L2158),
[HC8](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/kernels/cuda/hc8_gr.cu),
[MTP](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/core/mtp.cpp#L1225),
[IQ 프리필](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/prefill/moe_fused_iq.cu),
[Windows 지원 범위](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/docs/platforms.md),
[현재 Strata 측정 문서](DETAILS.md).

## 사용자 장비에서의 개선 가능성

W-2223이라는 CPU 모델 자체가 기능상 이식 장애는 아니다. CPU worker 수와 CPU/GPU 분담은
실측으로 정해야 한다. 긴 입력에서 KV가 자라면 expert의 GPU 상주 공간이 줄어들고,
CPU miss 처리가 늘 수 있다. 이때 GPU 커널만 빨라져도 전체 속도 개선은 작을 수 있다.
반대로 expert가 충분히 GPU에 남고 dense/MTP 검증이 큰 비중을 차지하면 이식의 가치가 커진다.

IQ3_XXS fixture의 expert 바이트 합은 42,912,972,800바이트, 약 39.97 GiB다.
이 값은 모델 전체 크기가 아니다. Dense, HC8, KV, MTP 상태, 임시 버퍼도 필요하므로
48 GB VRAM에 모든 것을 상주시킬 수 있다고 단정하지 않는다.
128 GB RAM은 문서상 구성 가능성을 뒷받침하지만, 실제 peak RAM/VRAM 및 Windows 메모리 등록을 확인해야 한다.
근거: [현재 expert fixture](../tests/data/native_experts/iq3_xxs.txt),
[Basalt KV·메모리 관리](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/docs/running.md).

또한 IQ3_XXS라는 파일 이름이어도 내부 tensor는 여러 양자화 형식을 섞어 쓴다.
한 형식의 커널만 이식해서는 이 모델 전용 지원도 완성되지 않는다.

실제 128K~256K 입력에서는 **첫 토큰까지의 시간과 이후 출력 속도를 별도로 평가**해야 한다.
출력만 2배 빨라져도 입력 처리가 대부분인 요청은 전체 대기 시간이 크게 줄지 않을 수 있다.
짧은 컨텍스트에서의 공개 속도를 이 장비의 긴 입력 속도로 환산하지 않는다.

Basalt 패키지에는 expert profile, HC8 사본, 튜닝된 MTP drafter도 들어 있다.
따라서 성능 차이를 전부 커널 차이라고 해석하면 안 된다. 동일한 본 모델 가중치 비교와
MTP 산출물의 차이를 기록하고, 커널 변경의 효과는 같은 산출물을 쓰는 A/B로 분리한다.
공식 IQ3_XXS `.basalt`를 재사용할 수 있다면 새 학습은 필요하지 않다.
[모델 파일 문서](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/docs/model-file.md).

## 최소 작업 범위: Strata의 IQ 프리필만

구현 상태: 코드와 패리티 테스트를 추가했고 Linux에서 CUDA 13.0, sm86 대상으로 두 실행 파일을 빌드했다.
GPU가 없어 테스트 실행은 `no CUDA device: skipped`로 끝났다. Windows sm86 빌드와 A6000 속도는 아직 확인하지 못했다.

요구 조건을 **최적화 개발 및 A6000 반복 시험 없이, 합리적인 개선 가능성을 가진 작은 변경**으로 좁히면
Basalt 전체 이식은 범위에서 빠진다. 현재 Strata에 이미 KV streaming, MTP,
CPU·PCIe 분담 자동 조정, sm80 이상 fused INT8 프리필이 있다.
우선 설치된 버전의 `START-HERE.bat --calibrate`와 서버 로그로 이 기능들이 실제 켜졌는지 확인한다.
128K~256K 문맥에서 기본 KV streaming을 유지한다. 설정 확인은 별도의 커널 개발 없이 수행 가능하다.
[현재 Strata 측정과 설정](DETAILS.md).

**작은 코드 변경 후보는 Basalt의 IQ3_XXS 프리필 v2 한 가지다.**
이 코드는 sm80 이상 `cp.async`·INT8 `mma.sync`를 사용하며, 필요한 공유 메모리는
gate/up 99,856바이트, down 98,832바이트다. 둘 다 sm86의 블록당 최대 99 KiB보다 작다.
기존 Strata 프리필 코드와 입력·출력 형식이 비슷하고 기존 커널로 되돌아갈 수 있다.
하지만 실제 A6000에서 성능이나 kernel launch 성공을 입증한 값은 없다.
[Basalt v2](https://github.com/jesdga95/basalt/blob/8d73a39477690e09f5fcc037bbe62f0a06481743/src/prefill/moe_fused_iq.cu#L560),
[NVIDIA sm86 공유 메모리 한도](https://docs.nvidia.com/cuda/archive/12.6.3/ampere-tuning-guide/index.html).

이 모델의 [48층 형식 fixture](../tests/data/native_experts/iq3_xxs.txt)를 세면 v2 대상은
gate/up IQ3_XXS(type 18) **6층**, down IQ4_NL(type 20) **18층**이다.
나머지 층은 기존 Strata 경로를 계속 사용한다. 따라서 이 패치만으로
Basalt의 전체 출력 속도 향상이나 2배 개선을 재현할 수 없다.
긴 입력의 **프리필 일부에 대한 제한된 개선 가능성**만 평가 대상이다.

구현 범위:

1. Strata의 `src/prefill/moe_fused_iq.cu`에 v2의 두 형식만 옮긴다.
   다른 형식과 HIP 경로를 변경하지 않는다.
2. `src/prefill/moe_fused.cu`의 그룹 작업 공간에 두 item counter용 64바이트를 추가하고,
   `include/strata/prefill/moe_fused_iq.hpp`에 필요한 선언을 추가한다.
3. 처음 구현은 `STRATA_PF_FUSED=1`과 별도 opt-in 스위치가 모두 있을 때만 v2를 썼다.
   이후 사용자 요청으로 sm86에서 IQ3_XXS gate/up 또는 IQ4_NL down을 사용하는 층은 기본값으로 켰다.
   `STRATA_PF_V2=0`은 기존 fused, `STRATA_PF_FUSED=0`은 MMQ 경로를 고른다.
   지원 검사나 커널 초기화가 실패하면 기존 fused 커널을 선택한다.
4. 기존 `tests/cuda/prefill_fused_iq_test.cpp`에 두 형식의 v2 대 기존 fused·참조 결과 비교를 추가한다.
   Windows sm86 컴파일과 가능한 Ampere GPU의 실행 결과를 검사한다.
5. A6000 접근이 단 한 번이라도 가능하면 128K·약 250K에서 v2 off/on 각각 한 번씩
   실행해 정상 출력·메모리·TTFT·prefill 속도를 기록한다. 이는 적합성 확인이며
   반복 측정이 아니므로 확정적인 성능 개선율로 제시하지 않는다.

**예상 작업량: 숙련된 CUDA 개발자 1명 기준 3~6 작업일, 프로덕션 파일 3개와 테스트 1개 내외.**
이는 Basalt v2 코드를 Strata에 맞추고 CUDA 빌드·기존 패리티 테스트를 확인하는 범위다.
Linux CUDA 13.0과 Windows MSVC sm86 빌드는 확인했다. GPU 실행은 아직 확인하지 못했다.
사용자 요청에 따라 A6000에서 기본값으로 켰지만 성능 개선이 검증됐다고 보고하지 않는다.

이 후보가 맞지 않으면 추가 커널 이식으로 범위를 넓히지 않는다.
현재 Strata의 설정 확인까지만 완료하고, 실측 가능한 시점에 다시 판단한다.
sm86 전용 dense/GDN/HC/MoE 이식, Basalt 모델 포맷 전환, NVFP4, 서버 개편은 이번 범위 밖이다.
그 일들은 A6000에서 병목을 확인한 뒤 별도 과제로 정해야 한다.

## 최소 범위의 완료 기준

- sm86 Windows 빌드가 되고, `STRATA_PF_V2=0`을 지정한 기존 fused 경로가 동작한다.
- 사용 가능한 Ampere GPU에서 gate/up type 18과 down type 20의 v2 결과를
  기존 fused 경로 및 참조 값과 비교한다. 수치 경로가 바뀌므로 품질 검사를 건너뛰지 않는다.
- A6000에서는 반복 튜닝 대신 가능할 때 한 차례의 긴 입력 적합성 확인을 한다.
  128K와 약 250K 입력에 출력 여유를 남기고, v2 off/on의 조건을 맞춰 기록한다.
- A6000 실행이 없다면 산출물은 **sm86 해당 형식에 기본 적용되는 미검증 패치**와 테스트 절차다.
  전체 요청 속도 개선은 아직 확인되지 않은 상태로 표기한다.

이 최소 범위는 성능 개선 가능성이 있는 한 경로를 낮은 유지비로 시험하는 수준이다.
기간 추정의 확신도는 중간 이하, A6000 개선율은 미정이다.
