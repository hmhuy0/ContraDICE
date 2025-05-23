import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.weight_decay = 0.1
    config.actor_Q_scale = 2.5
    config.v_beta = 15.0
    config.r_scale = 3.0
    config.r_alpha = 0.4
    config.tau = 0.005
    config.state_norm = False
 

    return config
