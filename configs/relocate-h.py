import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.weight_decay = 0.5
    config.actor_Q_scale = 0.2
    config.v_beta = 3.0
    config.r_scale = 3.0
    config.r_alpha = 0.8
    config.tau = 0.05
    config.state_norm = False

    return config
