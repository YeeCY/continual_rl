from src.agent.dqn_agent import DqnCnnSSEnsembleAgent
from src.agent.sac import EwcSacMlpAgent, SiSacMlpAgent, AgemSacMlpAgent, SacMlpAgent, \
    MultiHeadSacMlpAgent
from src.agent.ppo import PpoMlpAgent, EwcPpoMlpAgent, SiPpoMlpAgent, AgemPpoMlpAgent, \
    MultiHeadPpoMlpAgent, EwcMultiHeadPpoMlpAgent, SiMultiHeadPpoMlpAgent, AgemMultiHeadPpoMlpAgent

from agent.trash import SacMlpSSEnsembleAgent, SacCnnSSEnsembleAgent


ALGOS = [
    'dqn_cnn_ss_ensem',
    'sac_cnn_ss_ensem',
    'sac_mlp_ss_ensem',
    'sac_mlp',
    'mh_sac_mlp',
    'ewc_sac_mlp',
    'si_sac_mlp',
    'agem_sac_mlp',
    'ppo_mlp',
    'ewc_ppo_mlp',
    'si_ppo_mlp',
    'agem_ppo_mlp',
    'mh_ppo_mlp',
    'ewc_mh_ppo_mlp',
    'si_mh_ppo_mlp',
    'agem_mh_ppo_mlp',
]


def make_agent(obs_space, action_space, device, args):
    shape_attr = 'n' if args.env_type == 'atari' else 'shape'
    if isinstance(action_space, list):
        action_shape = [getattr(ac, shape_attr) for ac in action_space]
    else:
        action_shape = getattr(action_space, shape_attr)

    kwargs = {
        'obs_shape': obs_space.shape,
        'action_shape': action_shape,
        'discount': args.discount,
        # 'use_fwd': args.use_fwd,
        # 'use_inv': args.use_inv,
        # 'ss_lr': args.ss_lr,
        # 'ss_update_freq': args.ss_update_freq,
        # 'batch_size': args.batch_size,
        'device': device,
    }

    if args.algo == 'dqn_cnn_ss_ensem':
        kwargs['feature_dim'] = args.encoder_feature_dim
        kwargs['double_q'] = args.double_q
        kwargs['dueling'] = args.dueling
        kwargs['exploration_fraction'] = args.exploration_fraction
        kwargs['exploration_initial_eps'] = args.exploration_initial_eps
        kwargs['exploration_final_eps'] = args.exploration_final_eps
        kwargs['target_update_interval'] = args.target_update_interval
        kwargs['max_grad_norm'] = args.max_grad_norm
        kwargs['q_net_lr'] = args.q_net_lr
        kwargs['q_net_tau'] = args.q_net_tau
        kwargs['batch_size'] = args.batch_size

        agent = DqnCnnSSEnsembleAgent(**kwargs)
    elif 'sac' in args.algo:
        if isinstance(action_space, list):
            action_range = [[float(ac.low.min()),
                             float(ac.high.max())]
                            for ac in action_space]
        else:
            action_range = [float(action_space.low.min()),
                            float(action_space.high.max())]

        kwargs['action_range'] = action_range
        kwargs['hidden_dim'] = args.hidden_dim
        kwargs['init_temperature'] = args.init_temperature
        kwargs['alpha_lr'] = args.alpha_lr
        kwargs['actor_lr'] = args.actor_lr
        kwargs['actor_log_std_min'] = args.actor_log_std_min
        kwargs['actor_log_std_max'] = args.actor_log_std_max
        kwargs['actor_update_freq'] = args.actor_update_freq
        kwargs['critic_lr'] = args.critic_lr
        kwargs['critic_tau'] = args.critic_tau
        kwargs['critic_target_update_freq'] = args.critic_target_update_freq
        kwargs['batch_size'] = args.batch_size

        if args.algo == 'sac_cnn_ss_ensem':
            kwargs['use_fwd'] = args.use_fwd
            kwargs['use_inv'] = args.use_inv
            kwargs['ss_lr'] = args.ss_lr
            kwargs['ss_update_freq'] = args.ss_update_freq
            kwargs['num_ensem_comps'] = args.num_ensem_comps
            kwargs['encoder_feature_dim'] = args.encoder_feature_dim
            kwargs['encoder_lr'] = args.encoder_lr
            kwargs['encoder_tau'] = args.encoder_tau
            kwargs['ss_stop_shared_layers_grad'] = args.ss_stop_shared_layers_grad
            kwargs['num_layers'] = args.num_layers
            kwargs['num_shared_layers'] = args.num_shared_layers
            kwargs['num_filters'] = args.num_filters
            kwargs['curl_latent_dim'] = args.curl_latent_dim
            agent = SacCnnSSEnsembleAgent(**kwargs)
        elif args.algo == 'sac_mlp_ss_ensem':
            kwargs['action_range'] = [float(action_space.low.min()),
                                      float(action_space.high.max())]
            kwargs['use_fwd'] = args.use_fwd
            kwargs['use_inv'] = args.use_inv
            kwargs['ss_lr'] = args.ss_lr
            kwargs['ss_update_freq'] = args.ss_update_freq
            kwargs['num_ensem_comps'] = args.num_ensem_comps
            agent = SacMlpSSEnsembleAgent(**kwargs)
        elif args.algo == 'sac_mlp':
            agent = SacMlpAgent(**kwargs)
        elif args.algo == 'ewc_sac_mlp':
            kwargs['ewc_lambda'] = args.ewc_lambda
            kwargs['ewc_fisher_sample_size'] = args.ewc_fisher_sample_size
            kwargs['online_ewc'] = args.online_ewc
            kwargs['online_ewc_gamma'] = args.online_ewc_gamma
            agent = EwcSacMlpAgent(**kwargs)
        elif args.algo == 'si_sac_mlp':
            kwargs['si_c'] = args.si_c
            kwargs['si_epsilon'] = args.si_epsilon
            agent = SiSacMlpAgent(**kwargs)
        elif args.algo == 'agem_sac_mlp':
            kwargs['agem_memory_budget'] = args.agem_memory_budget
            kwargs['agem_ref_grad_batch_size'] = args.agem_ref_grad_batch_size
            agent = AgemSacMlpAgent(**kwargs)
        elif args.algo == 'mh_sac_mlp':
            agent = MultiHeadSacMlpAgent(**kwargs)
    elif 'ppo' in args.algo:
        kwargs['hidden_dim'] = args.ppo_hidden_dim
        kwargs['clip_param'] = args.ppo_clip_param
        kwargs['ppo_epoch'] = args.ppo_epoch
        kwargs['critic_loss_coef'] = args.ppo_critic_loss_coef
        kwargs['entropy_coef'] = args.ppo_entropy_coef
        kwargs['lr'] = args.ppo_lr
        kwargs['eps'] = args.ppo_eps
        kwargs['grad_clip_norm'] = args.ppo_grad_clip_norm
        kwargs['use_clipped_critic_loss'] = args.ppo_use_clipped_critic_loss
        kwargs['num_batch'] = args.ppo_num_batch

        if args.algo == 'ppo_mlp':
            agent = PpoMlpAgent(**kwargs)
        elif args.algo == 'ewc_ppo_mlp':
            kwargs['ewc_lambda'] = args.ppo_ewc_lambda
            kwargs['ewc_estimate_fisher_epochs'] = args.ppo_ewc_estimate_fisher_epochs
            kwargs['online_ewc'] = args.ppo_online_ewc
            kwargs['online_ewc_gamma'] = args.ppo_online_ewc_gamma
            agent = EwcPpoMlpAgent(**kwargs)
        elif args.algo == 'si_ppo_mlp':
            kwargs['si_c'] = args.ppo_si_c
            kwargs['si_epsilon'] = args.ppo_si_epsilon
            agent = SiPpoMlpAgent(**kwargs)
        elif args.algo == 'agem_ppo_mlp':
            kwargs['agem_memory_budget'] = args.ppo_agem_memory_budget
            kwargs['agem_ref_grad_batch_size'] = args.ppo_agem_ref_grad_batch_size
            agent = AgemPpoMlpAgent(**kwargs)
        elif args.algo == 'mh_ppo_mlp':
            agent = MultiHeadPpoMlpAgent(**kwargs)
        elif args.algo == 'ewc_mh_ppo_mlp':
            kwargs['ewc_lambda'] = args.ppo_ewc_lambda
            kwargs['ewc_estimate_fisher_epochs'] = args.ppo_ewc_estimate_fisher_epochs
            kwargs['online_ewc'] = args.ppo_online_ewc
            kwargs['online_ewc_gamma'] = args.ppo_online_ewc_gamma
            agent = EwcMultiHeadPpoMlpAgent(**kwargs)
        elif args.algo == 'si_mh_ppo_mlp':
            kwargs['si_c'] = args.ppo_si_c
            kwargs['si_epsilon'] = args.ppo_si_epsilon
            agent = SiMultiHeadPpoMlpAgent(**kwargs)
        elif args.algo == 'agem_mh_ppo_mlp':
            kwargs['agem_memory_budget'] = args.ppo_agem_memory_budget
            kwargs['agem_ref_grad_batch_size'] = args.ppo_agem_ref_grad_batch_size
            agent = AgemMultiHeadPpoMlpAgent(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm {args.algo}")

    return agent
