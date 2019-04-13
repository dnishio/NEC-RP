import math

from rl_coach.agents.dnec_agent import DNECAgentParameters
from rl_coach.architectures.tensorflow_components.heads.dueling_dnd_q_head import DuelingDNDQHeadParameters
from rl_coach.base_parameters import VisualizationParameters, MiddlewareScheme, PresetValidationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4, atari_schedule
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(10000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(99)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(2000)


#########
# Agent #
#########
agent_params = DNECAgentParameters()

# since we are using Adam instead of RMSProp, we adjust the learning rate as well
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].middleware_parameters.scheme = MiddlewareScheme.Shallow
agent_params.network_wrappers['main'].heads_parameters = \
    [DuelingDNDQHeadParameters(rescale_gradient_from_head_by_factor=1/math.sqrt(2))]
agent_params.network_wrappers['main'].clip_gradients = 10

###############
# Environment #
###############
env_params = Atari(level=SingleLevelSelection(atari_deterministic_v4))

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'pong', 'space_invaders']

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
