import torch
from itertools import chain

from agent.ppo.base_ppo_agent import PpoMlpAgent
from agent.network import MultiHeadPpoActorMlp, MultiHeadPpoCriticMlp


class MultiHeadPpoMlpAgent(PpoMlpAgent):
    def __init__(
            self,
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
            batch_size=32,
    ):
        assert isinstance(action_shape, list)
        super().__init__(
            obs_shape, action_shape, device, hidden_dim, discount, clip_param, ppo_epoch, critic_loss_coef,
            entropy_coef, lr, eps, grad_clip_norm, use_clipped_critic_loss, batch_size)

    def _setup_agent(self):
        if hasattr(self, 'actor') and hasattr(self, 'critic') \
                and hasattr(self, 'optimizer'):
            return

        self.actor = MultiHeadPpoActorMlp(
            self.obs_shape, self.action_shape, self.hidden_dim
        ).to(self.device)

        self.critic = MultiHeadPpoCriticMlp(
            self.obs_shape, self.hidden_dim, len(self.action_shape)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(chain(self.actor.parameters(), self.critic.parameters()),
                                          lr=self.lr, eps=self.eps)
