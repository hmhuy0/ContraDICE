import wandb

from dataclasses import dataclass
from .common import *
from .evaluation import evaluate

from pprint import pprint


@dataclass(frozen=True)
class ConfigArgs:
    max_clip: float
    v_beta: float
    eval_interval: int
    v_update: str
    adv_policy_extraction: bool
    r_scale: float
    r_alpha: float
    exp_r: bool
    learn_expert_data: bool
    good_reward_coeff: float
    bad_reward_coeff: float
        
def log_info(info,args,step):
    if (args.use_wandb):
        wandb.log(info, step=step)
    else:
        pprint(info) 