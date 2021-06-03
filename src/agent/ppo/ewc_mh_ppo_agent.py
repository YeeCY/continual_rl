from src.agent.ppo import MultiHeadPpoMlpAgent, EwcPpoMlpAgent


class EwcMultiHeadPpoMlpAgent(MultiHeadPpoMlpAgent, EwcPpoMlpAgent):
    """Adapt https://github.com/GMvandeVen/continual-learning"""
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 hidden_dim=64,
                 discount=0.99,
                 clip_param=0.2,
                 ppo_epoch=10,
                 critic_loss_coef=0.5,
                 entropy_coef=0.0,
                 lr=3e-4,
                 eps=1e-5,
                 grad_clip_norm=0.5,
                 use_clipped_critic_loss=True,
                 num_batch=32,
                 ewc_lambda=5000,
                 ewc_estimate_fisher_epochs=100,
                 online_ewc=False,
                 online_ewc_gamma=1.0,
                 ):
        MultiHeadPpoMlpAgent.__init__(self, obs_shape, action_shape, device, hidden_dim, discount, clip_param,
                                      ppo_epoch, critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm,
                                      use_clipped_critic_loss, num_batch)

        EwcPpoMlpAgent.__init__(self, obs_shape, action_shape, device, hidden_dim, discount, clip_param,
                                ppo_epoch, critic_loss_coef, entropy_coef, lr, eps, grad_clip_norm,
                                use_clipped_critic_loss, num_batch, ewc_lambda, ewc_estimate_fisher_epochs,
                                online_ewc, online_ewc_gamma)
