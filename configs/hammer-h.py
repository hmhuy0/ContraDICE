import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.weight_decay = 0.05
    config.actor_Q_scale = 0.2
    config.v_beta = 20.0
    config.r_scale = 3.0
    config.r_alpha = 0.6
    config.tau = 0.05
    config.state_norm = True
 

    return config
