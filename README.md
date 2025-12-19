# Bandit-over-Rankers

이 레포는 최적의 "랭킹 정책(=ranker) 선택"을 위한인 밴딧 알고리즘 시뮬레이터입니다.

## 주요 사항
- **Ranker(정책, arm)**: 후보 아이템을 점수화해서 순위를 만드는 규칙
- **Bandit(학습자)**: 매 step별로 컨텍스트를 보고 어떤 ranker를 쓸지 선택
- **cumulative regert (oracle)**: 매 시점 최적의 선택(Oracle)을 가정했을 때의 보상과 실제 획득 보상 간의 누적 차이
- **cumulative regret**: 최적 정책을 학습하는 과정에서 발생하는 누적 보상 손실

## 실행 방법 (MovieLens-10M 예시)
1. 우선 `data` 폴더 안에 `ml-10M`폴더를 만든 후 `ratings.dat`, `movies.dat`를 추가합니다.

2. 이후 아래를 실행하시면 `./outputs` 폴더가 생성되고 이 내부에 시드별 결과, 전체 종합 결과, reward/regret 그림이 생성됩니다.

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

![readme의 설정대로 실행할 경우의 reward / regret plot](assets/cumulative_alpha_comparison.png)


## 세부 데이터 집계

hit@k, mean_reward, switch_arm, topShare 정보를 표로 바로 산출하고 싶은 경우 아래 명령어를 사용합니다.

만일 가장 마지막 실험에 대한 산출을 원할 경우:
```
python3 summarize.py --check-dir "$(ls -dt .outputs/*/run_bandit_checks | head -n 1)"
```

`outputs`폴더의 특정 결과를 직접 지정해서 산출하고 싶은 경우:
```
python3 summarize.py --check-dir .outputs/{실행시간}/run_bandit_checks
```

각각 run_bandit_checks에 `aggregate_full.csv`, `aggregate_tail.csv`,  `aggregate_summary.json`파일이 생성됩니다.


## LLM 페르소나 임베딩 확장을 원할 경우
`--persona-npy path/to/user_persona.npy` 를 도입하면 (num_users, d_p) 임베딩을 밴딧의 컨텍스트에 concat하도록 설계되어 있습니다.
