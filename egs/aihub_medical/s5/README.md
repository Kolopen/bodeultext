# AI-Hub 의료진 및 환자 음성 → Kaldi 한국어 음성인식

AI-Hub **「의료진 및 환자 음성」** 데이터로 한국어 음성인식 모델을 학습하는 레시피다.
**Windows + NVIDIA GPU(RTX 5070)** 에서 WSL2 로 돌리는 것을 기준으로 적었다.
GMM 단계까지는 GPU 없이도 돌므로 Mac 에서도 된다(맨 아래).

```
라벨(JSON) + 음성(wav, 48kHz)
   │ local/aihub.py prep         전사 정규화, 16kHz 변환, Kaldi 데이터 폴더
   ▼
data/train, data/test
   │ local/morph.py              어절을 조각으로 (얼굴이 -> 얼굴 +이)
   │ local/prepare_dict.py       한글 표기에서 발음사전 (받침 대표음, 연음)
   │ local/train_lm.sh           3-gram 언어모델 (+ -AI 의료 용어)
   ▼
[CPU] mono -> tri1 -> tri2 -> tri3          run.sh 0~9 단계
[GPU] TDNN-F 신경망 (chain)                  run.sh 10 단계 = local/chain/run_tdnn.sh
   ▼
exp/chain/tdnn1a_online/  실제 녹음 인식에 쓰는 모델 묶음
```

## 왜 Whisper 대신 Kaldi 인가

-AI 저장소에서 확인한 Whisper 계열의 실패는 대부분 **디코더가 문장을 지어내는 것**이었다
(짧은 구간 환각, 용어 프롬프트에 끌려가 같은 말 반복). Kaldi 는 발음사전에 있는 단위의
조합만 내놓으므로 녹음에 없는 문장을 만들지 않는다. 의료 용어는 프롬프트가 아니라
발음사전과 언어모델에 직접 넣는다.

숫자는 Kaldi 도 "이백 십 육"처럼 **소리 그대로 한글로** 낸다. 학습 데이터가 소리를
적었기 때문이다. 정해진 규칙으로 216 으로 되돌릴 수 있으므로(후처리) 빠뜨리거나
지어내는 것보다 다루기 쉽다.

---

## 1. WSL2 준비 (Windows, 한 번만)

WSL2 는 Windows 안에 리눅스를 하나 더 띄운다. 기존 파일·프로그램은 건드리지 않고,
다 쓰면 명령 하나로 통째로 지운다(맨 아래 "다 쓰고 지우기").

**① NVIDIA 드라이버를 최신으로.** 평소 쓰는 Windows 드라이버(GeForce Experience /
NVIDIA 앱)를 업데이트만 하면 된다. WSL 은 이 드라이버를 그대로 쓴다. Windows 쪽에
CUDA 를 따로 깔지 않는다.

**② 외장 SSD 를 NTFS 로 포맷**(1TB 이상 권장). 리눅스 전체를 여기에 둬서 노트북
디스크를 쓰지 않는다. exFAT 은 안 된다.

**③ Ubuntu 설치 후 외장 SSD 로 옮기기** — 관리자 PowerShell (외장 SSD 가 `D:` 일 때)

```powershell
wsl --install -d Ubuntu-24.04          # 끝나면 재부팅, 사용자 이름/암호 만들기

wsl --shutdown
mkdir D:\wsl
wsl --export Ubuntu-24.04 D:\wsl\ubuntu.tar
wsl --unregister Ubuntu-24.04
wsl --import Ubuntu-24.04 D:\wsl\ubuntu D:\wsl\ubuntu.tar
del D:\wsl\ubuntu.tar
```

옮기고 나면 root 로 로그인된다. Ubuntu 안에서 원래 사용자로 바꿔 둔다.

```bash
sudo tee /etc/wsl.conf <<'EOF'
[user]
default=<처음 만든 사용자 이름>
EOF
```

**④ 메모리 한도.** WSL 은 기본으로 RAM 의 절반만 쓴다. `C:\Users\<이름>\.wslconfig`
파일을 만들어 늘린다(RAM 16GB 면 12GB, 32GB 면 24GB 정도).

```ini
[wsl2]
memory=12GB
```

PowerShell 에서 `wsl --shutdown` 후 Ubuntu 를 다시 열면 적용된다.

## 2. Ubuntu 안 준비 (한 번만)

여기부터는 전부 **Ubuntu 터미널**(시작 메뉴 → Ubuntu)에서 한다.

```bash
sudo apt update
sudo apt install -y build-essential git automake autoconf libtool sox gfortran \
  python3 python3-pip zlib1g-dev wget unzip unar subversion libopenblas-dev locales python-is-python3
sudo locale-gen en_US.UTF-8                       # 이게 없으면 Kaldi 가 한글을 깨진 글자로 본다
pip3 install --break-system-packages morfessor

nvidia-smi                                        # RTX 5070 이 보이면 GPU 연결 성공
```

**CUDA 툴킷 (12.8 이상).** RTX 5070 은 최신 세대(Blackwell)라 12.8 보다 낮으면 안 된다.
NVIDIA 의 WSL 전용 저장소에서 받는다(일반 리눅스용 드라이버는 깔지 않는다).

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
/usr/local/cuda/bin/nvcc --version                # 12.8 이 나오면 된다
```

## 3. Kaldi 빌드 (한 번만, 2~3시간)

```bash
git clone -b claude/eloquent-cray-29ssw9 https://github.com/Kolopen/bodeultext.git ~/bodeultext
git clone https://github.com/Kolopen/-AI.git ~/-AI          # 의료 용어 사전

cd ~/bodeultext/tools
extras/check_dependencies.sh          # MKL, python2.7 경고는 무시한다
make -j 4
OPENBLAS_TARGET=HASWELL extras/install_openblas.sh   # 행렬 계산 라이브러리 (최신 CPU 는 TARGET 지정 필요)

~/bodeultext/egs/aihub_medical/s5/local/configure_kaldi.sh   # 아래 configure 를 대신 실행
cd ../src
make depend -j 4
make -j 4
```

`-j 4` 는 WSL 메모리 12GB 기준이다. 메모리가 넉넉하면 늘린다.
`--mathlib=OPENBLAS` 를 빼면 x86 리눅스에서는 Intel MKL 을 찾다가 멈춘다.

`local/configure_kaldi.sh` 가 실제로 돌리는 명령은 이렇다. 터미널에 길게 붙여넣으면
줄이 깨져 `Unknown argument` 로 멈추는 일이 잦아 스크립트로 묶었다.

```bash
./configure --shared --use-cuda=yes --cudatk-dir=/usr/local/cuda \
  --cuda-arch="-gencode arch=compute_120,code=sm_120" \
  --mathlib=OPENBLAS --openblas-root=../tools/OpenBLAS/install
```

원래 Kaldi 의 configure 는 CUDA 12 에 gcc 12.3 미만만 허용해서 Ubuntu 24.04(gcc 13)에서
거절한다. 이 저장소의 configure 는 NVIDIA 지원표대로 CUDA 12.4+ 는 gcc 13, 12.8+ 는
gcc 14 까지 받도록 고쳐 두었다.

`--cuda-arch` 는 꼭 준다. Kaldi 의 기본 목록에 RTX 50 시리즈(sm_120)가 아직 없어서,
안 주면 빌드는 되는데 학습이 GPU 에서 돌지 않는다.

확인: `cd ../egs/yesno/s5 && ./run.sh` 끝에 `%WER 0.00` 이 나오면 된다.
에러가 나면 마지막 20줄을 공유해 주면 된다.

## 4. 데이터 받기

| 순서 | filekey | 내용 | 크기 |
|---|---|---|---|
| 1 | 48745, 48759 | 라벨링데이터 (Training, Validation) | 1.3GB |
| 2 | 48750, 48753 | Training 의사_1, 환자_1 | 38GB (약 110시간) |
| 2 | 48761, 48762 | Validation 의사_1, 환자_1 | 18GB |
| 3 | 나머지 | 결과를 보고 결정 | 약 230GB |

AI-Hub 에서 받은 파일은 겹겹이 싸여 있다. 받은 파일은 tar 이고(브라우저가 `환자1` 처럼
이름을 바꾸기도 한다), 그 안에 진짜 zip 의 1GB 조각들(`환자_1.zip.part0`,
`.part1073741824`, ...)이 들어 있다. `local/unpack_aihub.py` 가 조각들을 이어진 zip
하나처럼 읽어서 한 번에 처리한다. 임시 파일이 생기지 않는다.

```bash
cd ~/bodeultext/egs/aihub_medical/s5
# 음성: wav 를 바로 꺼낸다. 멈췄으면 같은 명령을 다시 하면 이어서 한다.
python3 local/unpack_aihub.py "/mnt/e/aihub-zip/받은파일" --out /mnt/e/aihub/train/audio
# 라벨: 풀지 않고 zip 하나로만 (local/aihub.py 가 zip 안을 바로 읽는다)
python3 local/unpack_aihub.py "/mnt/e/aihub-zip/받은라벨파일" --join /mnt/e/aihub/train/labels
```

받은 파일, 조각 폴더, 이미 합친 zip 어느 것을 줘도 된다. 파일마다 CRC 를 확인하고,
조각이 빠졌거나 다운로드가 덜 됐으면 알려준다. 끝에 `깨짐 0` 이면 받은 파일을 지운다.

`unar` 는 20GB 넘는 zip(zip64)을 중간까지만 풀고 `Archive parsing failed` 로 멈추는
일이 있어 쓰지 않는다. **폴더 이름에 공백이 없어야 한다**(Kaldi 가 경로를 공백으로 자른다).

## 5. 음성 받기 전에: 라벨 점검

라벨만 받은 상태에서 돌린다. 라벨은 **풀지 않고 zip 그대로** 줘도 된다(200만 개라 풀면 몇 시간 걸린다).
`--labels` 에는 폴더, zip 파일, zip 이 들어 있는 폴더를 모두 줄 수 있다. 전사에 숫자·영문·비식별화 태그가 얼마나 섞였는지,
몇 시간이 학습에 쓰일지 알려준다. **이 출력을 공유해 주면 정규화 규칙을 맞춘다.**

```bash
cd ~/bodeultext/egs/aihub_medical/s5
python3 local/aihub.py inspect --labels /mnt/e/aihub/train/labels
```

지금 규칙은 이렇다(`local/aihub.py` 의 `normalize`).

- 문장부호(`, . ? !`)와 간투어 표시(`아/`), 용어 표시(`*`)는 지운다.
- 이중전사 `(3)/(삼)` 은 소리 쪽 `삼` 을 쓴다.
- 비식별화 태그(`#@이름#`)가 있는 발화는 뺀다. 음성에는 실제 이름이 있어 글자와 어긋난다.
- 아라비아 숫자나 영문이 남은 발화는 일단 뺀다. 비율이 크면 읽기 규칙을 넣는다.

## 6. 학습

```bash
cd ~/bodeultext/egs/aihub_medical/s5
./run.sh --train-labels /mnt/e/aihub/train/labels --train-audio /mnt/e/aihub/train/audio \
         --test-labels  /mnt/e/aihub/valid/labels --test-audio  /mnt/e/aihub/valid/audio \
         --terms ~/-AI/src/voice_ai/data/terms
```

| 단계 | 내용 | 장치 | 110시간 기준 예상 |
|---|---|---|---|
| 0~4 | 데이터, 사전, 언어모델, 특징 | CPU | 1~2시간 |
| 5~9 | GMM (mono~tri3) + 평가 | CPU | 반나절 |
| 10 | 신경망 (i-vector, 격자, TDNN-F 4 epoch) | CPU+GPU | 반나절~하루 |

시간은 노트북 CPU 와 데이터 양에 따라 크게 다르다. 처음 돌릴 때 재 두면 다음 계획이 쉽다.

**멈췄다 이어 하기.** Ubuntu 창을 닫거나 노트북을 써야 하면 `Ctrl+C` 로 멈춘다.

```bash
./run.sh --stage 7 ...                       # GMM 은 단계 번호로 (run.sh 맨 위 참고)
local/chain/run_tdnn.sh --stage 12 --train-stage 37   # 신경망은 반복 번호로
```

신경망은 반복마다 모델을 저장한다. 이어 할 번호는 `exp/chain/tdnn1a/` 안의 가장 큰
`N.mdl` 의 N 이다.

**GPU 메모리가 모자라면**(`out of memory`) `local/chain/run_tdnn.sh --stage 12 --minibatch 64,32`.

**노트북 관리.** 충전기를 꽂고, 통풍되는 곳(쿨링패드)에 둔다. GPU 85°C 이하는 정상이다.
제조사 앱에 배터리 충전 80% 제한이 있으면 켜 둔다. 학습은 GPU 를 닳게 하지 않는다.

## 7. 결과 읽기

```
%WER ... exp/chain/tdnn1a/decode_test/...   어절 오류율
%WER ... (best_cer)                         글자 오류율  <- 이걸로 비교
```

한국어는 띄어쓰기가 흔들려 WER 이 실제보다 나빠 보인다. **CER 로 비교한다.**

**평가 문장.** 이 데이터는 약 4만 개 문장을 2,400여 명이 나눠 읽은 낭독체다(라벨 113만 개 중
99.9% 가 여러 번 읽힌 문장). 그래서 `run.sh` 는 Validation 중 **학습에 없던 문장만** 골라
평가한다(200개 미만이면 경고 후 전체). 언어모델도 문장마다 한 번씩만 넣는다.

**주의:** 이 데이터는 정해진 문장을 여러 사람이 읽은 낭독체다. 평가 문장이 학습 문장과
겹칠 수 있어 점수가 실제보다 좋게 나온다. 진짜 성능은 **-AI 의 실제 진료 녹음**으로
whisper turbo 와 같은 기준(검사 수치, 의료 용어, 지어낸 문장 수)으로 재야 한다.

## 다 쓰고 지우기

**① 남길 것 먼저 복사** (Ubuntu 안에서, 외장 SSD 가 `D:` 일 때). 수백 MB 다.

```bash
mkdir -p /mnt/d/kaldi-model
cd ~/bodeultext/egs/aihub_medical/s5
cp -rL exp/chain/tdnn1a_online exp/chain/tree/graph data/lang_test /mnt/d/kaldi-model/
```

**② 리눅스 통째로 삭제** — 관리자 PowerShell

```powershell
wsl --shutdown
wsl --unregister Ubuntu-24.04      # Kaldi, CUDA, 데이터, 학습 결과 전부 삭제
Remove-Item -Recurse -Force D:\wsl    # 남은 빈 폴더
Remove-Item $env:USERPROFILE\.wslconfig   # 메모리 설정 파일
```

**③ WSL 기능까지 끄기** (선택). 다시 쓸 일이 없으면.

```powershell
wsl --uninstall
dism /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
dism /online /disable-feature /featurename:VirtualMachinePlatform /norestart
```

재부팅하면 설치 전 상태다. 설정 → 앱 목록에 "Ubuntu" 가 남아 있으면 거기서 제거한다.
NVIDIA 드라이버는 원래 쓰던 것이라 그대로 둔다. Windows 쪽에는 CUDA 를 깔지 않았으므로
지울 것이 없다.

## Mac 에서 (GMM 만)

Mac 은 NVIDIA GPU 가 없어 10 단계(신경망)는 건너뛰고 tri3 에서 멈춘다.

```bash
brew install automake autoconf libtool sox wget cmake python git bash \
             coreutils gnu-sed gawk grep findutils unar
pip3 install morfessor
cd ~/bodeultext/tools && make -j 8
cd ../src && ./configure --shared && make depend -j 8 && make -j 8
```

`path.sh` 가 brew 의 GNU 도구를 앞에 둔다. 외장 SSD 는 APFS 로 포맷한다.

## 다음 작업

1. 실제 진료 녹음 하나를 인식하는 스크립트 (긴 녹음을 잘라 디코딩)
2. 한글 숫자 → 아라비아 숫자 후처리 ("이백 십 육" → 216)
3. -AI 녹음으로 whisper turbo 와 맞대기
