# atmaCup16 1st place solution

## Docker による環境構築

```sh
docker compose build
docker compose -f compose.cpu.yaml build

# bash に入る場合
docker compose run --rm kaggle bash  # GPU
docker compose -f compose.cpu.yaml run --rm kaggle-cpu bash # CPU

# jupyter lab を起動する場合
docker compose up  # GPU
docker compose -f compose.cpu.yaml up  # CPU
```

## スクリプトの実行方法

```sh
python experiments/check/run.py exp=001
python experiments/check/run.py exp=base
```

### Hydra による Config 管理
- 各スクリプトに共通する基本的な設定は yamls/config.yaml 内にある
- 各スクリプトによって変わる設定は、実行スクリプトのあるフォルダ(`{major_exp_name}`)の中に `exp/{minor_exp_name}.yaml` として配置することで管理。
    - 実行時に `exp={minor_exp_name}` で上書きする
    - `{major_exp_name}` と `{minor_exp_name}` の組み合わせで実験が再現できるようにする


## 再現実行方法
- 準備: input/atmaCup16_Dataset にデータを置く
- 環境：GCE n2-standard-48 (48 vCPU, 24 core, 192 GB memory)

```sh
# 実行環境
docker compose -f compose.cpu.yaml run --rm kaggle-cpu bash

# 候補・特徴量作成
python cand_unsupervised transition_prob_fix/run.py exp=base
python cand_unsupervised/transition_prob_all_fix/run.py exp=base
python cand_unsupervised/ranking_location/run.py exp=sml_cd
python cand_unsupervised/ranking_location/run.py exp=lrg_cd
python cand_unsupervised/ranking_location_all/run.py exp=sml_cd

python cand_unsupervised/split_transition_prob_fix/run.py exp=base
python cand_unsupervised/split_transition_prob_all_fix/run.py exp=base
python cand_unsupervised/split_transition_prob_bidirect_all_fix/run.py exp=base'
python cand_unsupervised/split_feat_transition_prob_location/run.py exp=base'

python cand_unsupervised/split_ranking/run.py exp=base
python cand_unsupervised/split_ranking_location/run.py exp=ken_cd
python cand_unsupervised/split_ranking_location/run.py exp=lrg_cd
python cand_unsupervised/split_ranking_location/run.py exp=sml_cd
python cand_unsupervised/split_ranking_location/run.py exp=wid_cd

# fold 作成
python generate_datasets/make_cv/run.py 

# 学習用データ生成
python generate_datasets/030_train_test_feat/run.py exp=base
python generate_datasets/030_train_test_feat/run.py exp=other
python generate_datasets/030_train_test_feat/run.py exp=first

# 学習：長さ1
python experiments/008_split/run.py exp=v030_first

# 学習：長さ1以外
python experiments/008_split/run.py exp=v030_other
python experiments/008_split/run.py exp=v030_other001
python experiments/012_cat_boost/run.py exp=v030_other_001

# 遷移行列の計算
python cand_supervised/prob_matrix_test_weight/run.py exp=021

# それぞれの予測値を結合してsubmissionファイルを作る
python experiments/ensemble_007/run.py exp=032
```
