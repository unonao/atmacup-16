# Kaggle テンプレート

## 特徴
- Docker によるポータブルでKaggleと同一の環境
- Hydra による実験管理
    - パスなど各スクリプトに共通する設定を yamls/config.yaml で管理
    - 実験用スクリプトファイルの変更を major バージョンとしてフォルダごとに管理
    - スクリプトごとの細かいパラメータ管理をminor バージョンとして同一フォルダ内に管理することでフォルダ移動の手間をなくす

## Structure
```text
.
├── .jupyter-settings: jupyter-lab の設定ファイル。compose.yamlでJUPYTERLAB_SETTINGS_DIRを指定している
├── Dockerfile
├── Dockerfile.cpu
├── LICENSE
├── README.md
├── compose.cpu.yaml
├── compose.yaml
├── exp
├── input
├── notebook
├── output
├── utils
└── yamls
```

## Docker による環境構築

```sh
docker compose build
docker compose -f compose.cpu.yaml build

# bash に入る場合
docker compose run --rm kaggle bash 
docker compose -f compose.cpu.yaml run --rm kaggle-cpu bash

# jupyter lab を起動する場合
docker compose up 
docker compose -f compose.cpu.yaml up 
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


### 候補生成
```sh
 python cand_unsupervised/ranking/run.py
 python cand_unsupervised/ranking_location/run.py exp=sml_cd
 python cand_unsupervised/ranking_location/run.py exp=lrg_cd
 python cand_unsupervised/ranking_location/run.py exp=ken_cd
 python cand_unsupervised/ranking_location/run.py exp=wid_cd
 python cand_unsupervised/transition_prob/run.py

 python cand_unsupervised/ranking_location_all/run.py exp=sml_cd
```

### 学習データ生成
```sh
python generate_datasets/make_cv/run.py 
python generate_datasets/002_add_features/run.py 
```

### 学習&推論
```sh
python experiments/005_one_epoch/run.py exp=001 exp.one_epoch=True
python experiments/005_one_epoch/run.py exp=v2_004
python experiments/005_one_epoch/run.py exp=v2_005
```