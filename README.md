# ContraDICE


## installation
1. ```conda create -n contradice python=3.9```
2. run the following:
```
conda activate contradice
conda install nvidia/label/cuda-12.3.2::cuda-toolkit
conda install -c conda-forge cudnn==8.9.7.29
pip install -U "jax[cuda12]"==0.4.28

conda install -c conda-forge mesalib
conda install -c conda-forge glew
conda install -c menpo glfw3

export CPATH=$CONDA_PREFIX/include
pip install patchelf

pip install d4rl
pip install git+https://github.com/aravindr93/mjrl@master#egg=mjrl
pip install -U -r requirements.txt
```

## example scripts

The script below is for CHEETAH (RANDOM + EXPERT) with:

1. one single EXPERT trajectory in good dataset
2. 10 random trajectories in bad dataset
3. combine full (or -1 in the bash script) random trajectories and 30 expert trajectories for the unlabeled dataset.
4. `is_good_list` and `is_bad_list` is for debug and logging purpose.

```
#####    Mujoco halfcheetah (random + expert)     #####
# one single EXPERT trajectory in good dataset
# 10 RANDOM trajectories in bad dataset
# combine full (or -1) RANDOM trajectories 
# and 30 EXPERT trajectories for the unlabeled dataset.
##################################################

python train.py \
--env_name=halfcheetah-expert-v2 --max_steps=1000000 \
--expert_dataset_size=1 \
--bad_name_list=random \
--bad_size_list=10 \
--mixed_name_list=random,expert \
--mixed_size_list=-1,30 \
--is_good_list=0,1 \
--is_bad_list=1,0 \
--exp_name=test --seed=0
```

```
#####    Adroit Hammer (cloned + expert)     #####
# one single EXPERT trajectory in good dataset
# 25 CLONED trajectories in bad dataset
# combine full (or -1) CLONED trajectories 
# and 100 EXPERT trajectories for the unlabeled dataset.
##################################################

python train.py \
--env_name=hammer-expert-v1 --max_steps=1000000 \
--expert_dataset_size=1 \
--bad_name_list=cloned \
--bad_size_list=25 \
--mixed_name_list=cloned,expert \
--mixed_size_list=-1,100 \
--is_good_list=0,1 \
--is_bad_list=1,0 \
--exp_name=test --seed=0
```

```
#####    Kitchen (partial + complete)      #####
# one single COMPLETE trajectory in good dataset
# 25 PARTIAL trajectories in bad dataset
# combine full (or -1) PARTIAL trajectories 
# and 1 COMPLETE trajectories for the unlabeled dataset.
##################################################

python train.py \
--env_name=Kitchen-complete-v0 --max_steps=1000000 \
--expert_dataset_size=1 \
--bad_name_list=partial \
--bad_size_list=25 \
--mixed_name_list=partial,complete \
--mixed_size_list=-1,1 \
--is_good_list=0,1 \
--is_bad_list=1,0 \
--exp_name=test --seed=0
```

## Directory tree

```
.
├── configs
├── README.md
├── requirements.txt
├── run.sh
├── sources
│   ├── algos
│   │   └── ContraDICE
│   │       ├── actor.py
│   │       ├── algo.py
│   │       ├── critic.py
│   │       ├── disc.py
│   ├── dataset
│   │   ├── d4rl_dataset.py
│   │   ├── mix_dataset.py
│   ├── networks
│   │   ├── critic.py
│   │   ├── discriminator.py
│   │   ├── policy.py
│   │   └── value.py
│   ├── parse.py
│   └── utils
│       ├── common.py
│       ├── env_wrappers.py
│       ├── evaluation.py
│       └── __init__.py
└── train.py
```