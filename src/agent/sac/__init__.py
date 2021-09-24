from agent.sac.base_sac_agent import SacMlpAgent
from agent.sac.ewc_sac_agent import EwcSacMlpAgent
from agent.sac.ewc_sac_agent_v2 import EwcSacMlpAgentV2
from agent.sac.ewc_v2_sac_agent_v2 import EwcV2SacMlpAgentV2
from agent.sac.si_sac_agent import SiSacMlpAgent
from agent.sac.si_sac_agent_v2 import SiSacMlpAgentV2
from agent.sac.agem_sac_agent import AgemSacMlpAgent
from agent.sac.agem_sac_agent_v2 import AgemSacMlpAgentV2
from agent.sac.agem_continual_actor_sac_agent import AgemV2SacMlpAgentV2
from agent.sac.oracle_agem_v2_sac_agent_v2 import OracleAgemV2SacMlpAgentV2
from agent.sac.oracle_grad_agem_v2_sac_agent_v2 import OracleGradAgemV2SacMlpAgentV2
from agent.sac.oracle_actor_agem_v2_sac_agent_v2 import OracleActorAgemV2SacMlpAgentV2
from agent.sac.mh_sac_agent import MultiHeadSacMlpAgent
from agent.sac.mh_sac_agent_v2 import MultiHeadSacMlpAgentV2
from agent.sac.individual_sac_agent_v2 import IndividualSacMlpAgentV2
from agent.sac.mi_sac_agent_v2 import MultiInputSacMlpAgentV2
from agent.sac.ewc_mh_sac_agent import EwcMultiHeadSacMlpAgent
from agent.sac.ewc_mh_sac_agent_v2 import EwcMultiHeadSacMlpAgentV2
from agent.sac.ewc_v2_mh_sac_agent_v2 import EwcV2MultiHeadSacMlpAgentV2
from agent.sac.ewc_v2_mi_sac_agent_v2 import EwcV2MultiInputSacMlpAgentV2
from agent.sac.si_mh_sac_agent import SiMultiHeadSacMlpAgent
from agent.sac.si_mh_sac_agent_v2 import SiMultiHeadSacMlpAgentV2
from agent.sac.si_mi_sac_agent_v2 import SiMultiInputSacMlpAgentV2
from agent.sac.agem_mh_sac_agent import AgemMultiHeadSacMlpAgent
from agent.sac.agem_mh_sac_agent_v2 import AgemMultiHeadSacMlpAgentV2
from agent.sac.agem_mi_sac_agent_v2 import AgemMultiInputSacMlpAgentV2
from agent.sac.agem_continual_actor_mh_sac_agent import AgemV2MultiHeadSacMlpAgentV2
from agent.sac.oracle_agem_v2_mh_sac_agent_v2 import OracleAgemV2MultiHeadSacMlpAgentV2
from agent.sac.oracle_grad_agem_v2_mh_sac_agent_v2 import OracleGradAgemV2MultiHeadSacMlpAgentV2
from agent.sac.oracle_actor_agem_v2_mh_sac_agent_v2 import OracleActorAgemV2MultiHeadSacMlpAgentV2
from agent.sac.agem_continual_actor_mi_sac_agent import AgemV2MultiInputSacMlpAgentV2
from agent.sac.oracle_agem_v2_mi_sac_agent_v2 import OracleAgemV2MultiInputSacMlpAgentV2
from agent.sac.oracle_grad_agem_v2_mi_sac_agent_v2 import OracleGradAgemV2MultiInputSacMlpAgentV2
from agent.sac.oracle_actor_agem_v2_mi_sac_agent_v2 import OracleActorAgemV2MultiInputSacMlpAgentV2
from agent.sac.fisher_brc_mt_bc_mlp_critic_mh_sac_agent import FisherBRCMTBCMlpCriticMultiHeadSacMlpAgent
from agent.sac.fisher_brc_mt_bc_offset_critic_mh_sac_agent import FisherBRCMTBCOffsetCriticMultiHeadSacMlpAgent
from agent.sac.fisher_brc_mh_bc_mlp_critic_mh_sac_agent import FisherBRCMHBCMlpCriticMultiHeadSacMlpAgent
from agent.sac.fisher_brc_mh_bc_offset_critic_mh_sac_agent import FisherBRCMHBCOffsetCriticMultiHeadSacMlpAgent
from agent.sac.ewc_v2_grad_norm_reg_critic_mh_sac_agent_v2 import EwcV2GradNormRegCriticMultiHeadSacMlpAgentV2
from agent.sac.ewc_v2_grad_norm_reg_critic_mi_sac_agent_v2 import EwcV2GradNormRegCriticMultiInputSacMlpAgentV2
from agent.sac.agem_continual_actor_grad_norm_reg_critic_mh_sac_agent import \
    AgemV2GradNormRegCriticMultiHeadSacMlpAgentV2

from agent.sac.mi_sac_agent import MultiInputSacMlpAgent
from agent.sac.agem_continual_actor_critic_sac_agent import AgemContinualActorCriticSacMlpAgent
from agent.sac.agem_continual_actor_critic_mh_sac_agent import AgemContinualActorCriticMultiHeadSacMlpAgent
from agent.sac.agem_continual_actor_critic_mi_sac_agent import AgemContinualActorCriticMultiInputSacMlpAgent
from agent.sac.agem_continual_actor_critic_grad_norm_reg_critic_mh_sac_agent import \
    AgemContinualActorCriticGradNormRegCriticMultiHeadSacMlpAgent
from agent.sac.agem_continual_actor_critic_grad_norm_reg_critic_mi_sac_agent import \
    AgemContinualActorCriticGradNormRegCriticMultiInputSacMlpAgent
from agent.sac.agem_continual_actor_critic_grad_norm_reg_critic_prioritized_memory_mh_sac_agent import \
    AgemContinualActorCriticGradNormRegCriticPrioritizedMemoryMultiHeadSacMlpAgent
from agent.sac.agem_continual_actor_critic_grad_norm_reg_critic_prioritized_memory_mi_sac_agent import \
    AgemContinualActorCriticGradNormRegCriticPrioritizedMemoryMultiInputSacMlpAgent

from agent.sac.distilled_actor_mh_sac_agent import \
    DistilledActorMultiHeadSacMlpAgent
from agent.sac.distilled_actor_mi_sac_agent import \
    DistilledActorMultiInputSacMlpAgent

from agent.sac.task_embedding_hypernet_actor_sac_agent import \
    TaskEmbeddingHyperNetActorSacMlpAgent
from agent.sac.ewc_task_embedding_hypernet_actor_sac_agent import \
    EwcTaskEmbeddingHyperNetActorSacMlpAgent
from agent.sac.si_task_embedding_hypernet_actor_sac_agent import \
    SiTaskEmbeddingHyperNetActorSacMlpAgent
from agent.sac.agem_task_embedding_hypernet_actor_sac_agent import \
    AgemTaskEmbeddingHyperNetActorSacMlpAgent

from agent.sac.task_embedding_distilled_actor_sac_agent import \
    TaskEmbeddingDistilledActorSacMlpAgent
from agent.sac.ewc_task_embedding_distilled_actor_sac_agent import \
    EwcTaskEmbeddingDistilledActorSacMlpAgent
