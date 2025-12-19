# Bandit-over-Rankers

이 레포는 최적의 "랭킹 정책(=ranker) 선택"을 위한인 밴딧 알고리즘 시뮬레이터입니다.

## 주요 사항
- **Ranker(정책, arm)**: 후보 아이템을 점수화해서 순위를 만드는 규칙
- **Bandit(학습자)**: 매 step별로 컨텍스트를 보고 어떤 ranker를 쓸지 선택
- **cumulative regert (oracle)**: 매 시점 최적의 선택(Oracle)을 가정했을 때의 보상과 실제 획득 보상 간의 누적 차이
- **cumulative regret**: 최적 정책을 학습하는 과정에서 발생하는 누적 보상 손실

## 실행 (MovieLens-10M 예시)
데이터 폴더에 `ratings.dat`, `movies.dat`가 있어야 합니다.

```bash
pip install -r requirements.txt

python3 -m bandit_ranker_m1.cli \
  --loader movielens10m \
  --data-dir ./bandit_ranker/data/ml-10M \
  --n-steps 100000 \
  --n-candidates 250 \
  --k 5 \
  --seed 42 \
  --start-index 100000 \
  --n-seeds 5 \
  --user-profile True \
```

## 출력
- `outputs/stepwise.csv` : step별 reward, 상한 대비 격차, 고정랭커 대비 regret, hit@K 등
- `outputs/summary.json` : 최종 평균/누적 지표 및 best-fixed ranker 정보

## 데이터 교체
코어는 (user_id, item_id, timestamp, label) 이벤트 로그만 요구합니다.
MovieLens는 loader 예시일 뿐이며, 다른 데이터셋도 같은 인터페이스로 교체 가능합니다.

## 세부 데이터 집계
만일 가장 마지막 실험의 hit@k, mean_reward, switch_arm, topShare 정보를 표로 산출하고 싶은 경우 아래를 실행합니다.
```
python3 summarize.py --check-dir "$(ls -dt .outputs/*/run_bandit_checks | head -n 1)"
```

특정 폴더를 직접 지정해서 산출하고 싶은 경우:
```
python3 summarize.py --check-dir .outputs/{실행시간}/run_bandit_checks
```

각각 run_bandit_checks에 `aggregate_full.csv`, `aggregate_tail.csv`,  `aggregate_summary.json`파일이 생성됩니다.

## LLM 페르소나 임베딩 확장
`--persona-npy path/to/user_persona.npy` 를 주면 (num_users, d_p) 임베딩을 컨텍스트에 concat하도록 설계되어 있습니다.
