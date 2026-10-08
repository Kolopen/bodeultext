# AI-Hub 의료진 및 환자 음성 → Kaldi 한국어 음성인식 (Mac)

AI-Hub **「의료진 및 환자 음성」** 데이터로 한국어 음성인식 모델을 학습하는 레시피다.
GPU 없이 **Mac(M4, 16GB)** 에서 GMM 모델까지 끝까지 돈다. 신경망 단계는 이 결과를
보고 붙인다.

```
라벨(JSON) + 음성(wav, 48kHz)
   │ local/aihub.py prep         전사 정규화, 16kHz 변환, Kaldi 데이터 폴더
   ▼
data/train, data/test
   │ local/morph.py              어절을 조각으로 (얼굴이 -> 얼굴 +이)
   │ local/prepare_dict.py       한글 표기에서 발음사전 (받침 대표음, 연음)
   │ local/train_lm.sh           3-gram 언어모델 (+ -AI 의료 용어)
   ▼
mono -> tri1 -> tri2 (LDA+MLLT) -> tri3 (SAT)  ->  평가 (WER, CER)
```

## 왜 Whisper 대신 Kaldi 인가

-AI 저장소에서 확인한 Whisper 계열의 실패는 대부분 **디코더가 문장을 지어내는 것**이었다
(짧은 구간 환각, 용어 프롬프트에 끌려가 같은 말 반복). Kaldi 는 발음사전에 있는 단위의
조합만 내놓으므로 녹음에 없는 문장을 만들지 않는다. 의료 용어는 프롬프트가 아니라
발음사전과 언어모델에 직접 넣는다.

숫자는 Kaldi 도 "이백 십 육"처럼 **소리 그대로 한글로** 낸다. 학습 데이터가 소리를
적었기 때문이다. 이건 정해진 규칙으로 216 으로 되돌릴 수 있으므로(후처리) 빠뜨리거나
지어내는 것보다 다루기 쉽다.

## 1. Mac 준비 (한 번만)

```bash
xcode-select --install
# Homebrew: https://brew.sh
brew install automake autoconf libtool sox wget cmake python git bash \
             coreutils gnu-sed gawk grep findutils unar
pip3 install morfessor
```

macOS 기본 `sed`, `sort` 등은 Kaldi 스크립트와 동작이 달라서 GNU 판을 같이 깐다.
`path.sh` 가 알아서 앞에 둔다.

**외장 SSD(1TB 이상)를 APFS 로 포맷**해서 데이터와 학습 결과를 둔다. exFAT 은
Kaldi 가 쓰는 심볼릭 링크를 못 만들어 안 된다.

## 2. Kaldi 빌드 (한 번만, 1시간 안팎)

```bash
git clone https://github.com/Kolopen/bodeultext.git ~/bodeultext
cd ~/bodeultext/tools
extras/check_dependencies.sh     # 빠진 것이 있으면 알려준다
make -j 8

cd ../src
./configure --shared             # Mac 은 Apple Accelerate 를 쓴다
make depend -j 8
make -j 8
```

확인: `cd ../egs/yesno/s5 && ./run.sh` 끝에 `%WER 0.00` 이 나오면 된다.

## 3. 데이터 받기

[aihubshell](https://www.aihub.or.kr/static/pdf/aihubshell_가이드.pdf) 로 받는다.
브라우저 다운로드는 수십 GB 에서 잘 끊긴다. 명령 형식은 가이드와 `./aihubshell -help`
기준으로 확인한다.

```bash
mkdir -p /Volumes/SSD/aihub && cd /Volumes/SSD/aihub
curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x aihubshell
./aihubshell -mode l | grep 의료진                         # datasetkey 확인
./aihubshell -mode d -datasetkey <번호> -filekey 48745,48759 -aihubapikey '<키>'
```

| 순서 | filekey | 내용 | 크기 |
|---|---|---|---|
| 1 | 48745, 48759 | 라벨링데이터 (Training, Validation) | 1.3GB |
| 2 | 48750, 48753 | Training 의사_1, 환자_1 | 38GB (약 110시간) |
| 2 | 48761, 48762 | Validation 의사_1, 환자_1 | 18GB |
| 3 | 나머지 | 결과를 보고 결정 | 약 230GB |

압축은 `unar` 로 푼다. 기본 `unzip` 은 한글 파일 이름이 깨진다. **폴더 이름에 공백이
없어야 한다**(Kaldi 가 경로를 공백으로 자른다).

```bash
unar -o train/labels 라벨링데이터.zip
unar -o train/audio  의료진_의사_1.zip
unar -o train/audio  환자_1.zip
```

## 4. 음성 받기 전에: 라벨 점검

라벨만 받은 상태에서 돌린다. 전사에 숫자·영문·비식별화 태그가 얼마나 섞였는지,
몇 시간이 학습에 쓰일지 알려준다. **이 출력을 공유해 주면 정규화 규칙을 맞춘다.**

```bash
cd ~/bodeultext/egs/aihub_medical/s5
python3 local/aihub.py inspect --labels /Volumes/SSD/aihub/train/labels
```

지금 규칙은 이렇다(`local/aihub.py` 의 `normalize`).

- 문장부호(`, . ? !`)와 간투어 표시(`아/`), 용어 표시(`*`)는 지운다.
- 이중전사 `(3)/(삼)` 은 소리 쪽 `삼` 을 쓴다.
- 비식별화 태그(`#@이름#`)가 있는 발화는 뺀다. 음성에는 실제 이름이 있어 글자와 어긋난다.
- 아라비아 숫자나 영문이 남은 발화는 일단 뺀다. 비율이 크면 읽기 규칙을 넣는다.

## 5. 학습

학습 결과(`data`, `exp`, `mfcc`)는 수십 GB 가 되므로 외장 SSD 에 두고 링크한다.

```bash
cd ~/bodeultext/egs/aihub_medical/s5
mkdir -p /Volumes/SSD/kaldi-work/{data,exp,mfcc}
ln -s /Volumes/SSD/kaldi-work/data data
ln -s /Volumes/SSD/kaldi-work/exp exp
ln -s /Volumes/SSD/kaldi-work/mfcc mfcc

./run.sh --train-labels /Volumes/SSD/aihub/train/labels \
         --train-audio  /Volumes/SSD/aihub/train/audio \
         --test-labels  /Volumes/SSD/aihub/valid/labels \
         --test-audio   /Volumes/SSD/aihub/valid/audio \
         --terms ~/-AI/src/voice_ai/data/terms
```

멈췄으면 `--stage N` 으로 이어서 한다(단계 번호는 `run.sh` 맨 위). 학습 중에는
잠자기를 막는다: 다른 터미널에서 `caffeinate -i` 를 켜 두거나 `caffeinate -i ./run.sh ...`.

110시간 기준 대략 반나절~하루를 예상한다. 메모리가 모자라 멈추면 `--nj 4` 로 줄인다.

## 6. 결과 읽기

```
%WER ... exp/tri3/decode_test/...   어절 오류율
%WER ... (best_cer)                 글자 오류율  <- 이걸로 비교
```

한국어는 띄어쓰기가 흔들려 WER 이 실제보다 나빠 보인다. **CER 로 비교한다.**

**주의:** 이 데이터는 정해진 문장을 여러 사람이 읽은 낭독체다. 평가 문장이 학습 문장과
겹칠 수 있어 점수가 실제보다 좋게 나온다. 진짜 성능은 **-AI 의 실제 진료 녹음**으로
whisper turbo 와 같은 기준(검사 수치, 의료 용어, 지어낸 문장 수)으로 재야 한다.
그 비교 스크립트가 다음 작업이다.

## 다음 작업

1. 실제 진료 녹음 하나를 tri3 로 인식하는 스크립트 (긴 녹음을 잘라 디코딩)
2. 한글 숫자 → 아라비아 숫자 후처리 ("이백 십 육" → 216)
3. -AI 녹음으로 whisper turbo 와 맞대기
4. 결과가 좋으면 데이터를 늘리고 신경망(chain) 단계 추가
