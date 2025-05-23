from absl import flags
from ml_collections import config_flags


args = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('v_update', 'rkl_loss','Value update function.')
flags.DEFINE_string('exp_name', 'dump', 'Epoch logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_boolean('use_wandb', False, 'Use wandb')
flags.DEFINE_boolean('adv_policy_extraction', False, 'Use adv policy extraction')
flags.DEFINE_boolean('learn_expert_data', False, 'Learn expert dataset')

flags.DEFINE_boolean('exp_r', False, 'Use exp reward')
flags.DEFINE_integer('max_episode_steps', None, 'Max episode steps')

# general paramters
flags.DEFINE_float('actor_lr', 3e-4, 'Actor learning rate')
flags.DEFINE_float('value_lr', 3e-4, 'Value learning rate')
flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate')
flags.DEFINE_float('disc_lr', 3e-4, 'Discriminator learning rate')

flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate')
flags.DEFINE_boolean('layernorm', False, 'Use layernorm')
flags.DEFINE_float('discount', 0.99, 'Discount factor')

flags.DEFINE_integer('hidden_size', 256, 'Hidden size')
flags.DEFINE_integer('num_layers', 2, 'Number of layers')

# offline IL only

flags.DEFINE_integer('num_disc_train', int(1e5), 'Number of discriminator train steps')

flags.DEFINE_list('bad_name_list', None, 'List of bad dataset names')
flags.DEFINE_list('bad_size_list', None, 'List of bad dataset sizes')

flags.DEFINE_integer('expert_dataset_size', None, 'Expert dataset size')

flags.DEFINE_list('mixed_name_list', None, 'List of mixed dataset names')
flags.DEFINE_list('mixed_size_list', None, 'List of mixed dataset sizes')
flags.DEFINE_list('is_good_list', None, 'List of good dataset names')
flags.DEFINE_list('is_bad_list', None, 'List of bad dataset names')

