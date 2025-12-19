# Offline Bandit Rankers

이 레포는 최적의 랭킹 정책(=ranker) 선택을 위한 밴딧 알고리즘 시뮬레이터입니다.
<br>
## 주요 사항
- **Ranker(정책, arm)**: 후보 아이템을 점수화해서 순위를 만드는 규칙
- **Bandit(학습자)**: 매 step별로 컨텍스트를 보고 어떤 ranker를 쓸지 선택
- **cumulative regert (oracle)**: 매 시점 최적의 선택(Oracle)을 가정했을 때의 보상과 실제 획득 보상 간의 누적 차이
- **cumulative regret**: 최적 정책을 학습하는 과정에서 발생하는 누적 보상 손실
<br>

## 실행 방법 (MovieLens-10M 예시)

**1. 데이터 준비**

이 프로젝트는 MovieLens-10M 원본 데이터(ratings.dat, movies.dat)를 레포에 포함하지 않습니다(용량 제한).

다운로드 후, 아래 권장 위치대로 파일을 다운받은 후 ml-10M 폴더 경로를 --data-dir로 지정합니다.

예시(권장 디폴트 위치):

`./data/ml-10M/ratings.dat`

`./data/ml-10M/movies.dat`
<br>

**2. 실행**

```bash
pip install -r requirements.txt

python3 run_cli.py \
  --loader movielens10m \
  --data-dir ./data/ml-10M \
  --n-steps 100000 \
  --n-candidates 250 \
  --k 5 \
  --seed 42 \
  --start-index 100000 \
  --n-seeds 5 \
  --user-profile True
```
실행이 끝나면 .outputs/timestamp/ 아래에 시드별 결과 JSON, 집계 결과, reward/regret plot이 저장됩니다.

![readme의 설정대로 실행할 경우의 reward / regret plot](assets/cumulative_alpha_comparison.png)

<br>

## 세부 결과 집계

`./outputs/timestamp/run_bandit_checks`에 저장된 seed별 JSON을 집계하여, **bandit별 hit@k, reward의 means, std 표**와 **switch_arm, topShare 등 요약 JSON 파일**을 생성합니다.

- 생성 파일:
  - `aggregate_full.csv` (전체 구간 평균/표준편차)
  - `aggregate_tail.csv` (마지막 20% tail 구간 평균/표준편차)
  - `aggregate_summary.json` (메타정보 및 전체 집계)

**가장 최근 실행 결과를 집계**
  
```
python3 summarize.py --check-dir "$(ls -dt .outputs/*/run_bandit_checks | head -n 1)"
```

**`outputs`폴더의 특정 실행(timestamp) 결과를 집계**
```
python3 summarize.py --check-dir .outputs/{timestamp}/run_bandit_checks
```

각각 run_bandit_checks에 `aggregate_full.csv`, `aggregate_tail.csv`,  `aggregate_summary.json`파일이 생성됩니다.

<br>

## LLM 등 유저 페르소나 임베딩 확장
실행 명령어에서 `--persona-npy path/to/user_persona.npy`를 통해 밴딧의 컨텍스트와 추가 임베딩이 결합되도록 하였습니다.
<br>
