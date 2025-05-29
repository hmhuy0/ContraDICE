import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.weight_decay = 0.0
    config.actor_Q_scale = 1.5
    config.v_beta = 20.0
    config.r_scale = 2.0
    config.r_alpha = 0.3
    config.tau = 0.005
    config.state_norm = True
 
    
    return config
