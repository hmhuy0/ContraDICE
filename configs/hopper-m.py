import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.weight_decay = 0.001
    config.actor_Q_scale = 2.0
    config.v_beta = 20.0
    config.r_scale = 3.0
    config.r_alpha = 0.4
    config.tau = 0.005
    config.state_norm = True
 
    
    return config
