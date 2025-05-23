import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.weight_decay = 0.1
    config.actor_Q_scale = 1.5
    config.v_beta = 30.0
    config.r_scale = 5.0
    config.r_alpha = 0.3
    config.tau = 0.03
    config.state_norm = False
 
    
    return config
