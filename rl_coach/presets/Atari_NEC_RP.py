from rl_coach.agents.nec_agent import NECAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, MiddlewareScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, AtariInputFilter, atari_deterministic_v4
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.architectures.tensorflow_components.layers import RPDense


####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(10000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(50000)

#########
# Agent #
#########

agent_params = NECAgentParameters(scheme=MiddlewareScheme.RP, dense_layer=RPDense)
# If the rp_to_fc_step is inf, use only RP layers. Also if it's 0 or None, use FC layers only (same as NEC).
agent_params.rp_to_fc_step = 100000000
agent_params.network_wrappers['main'].learning_rate = 0.00001
agent_params.input_filter = AtariInputFilter()
agent_params.input_filter.remove_reward_filter('clipping')

###############
# Environment #
###############
env_params = Atari(level=SingleLevelSelection(atari_deterministic_v4))
env_params.random_initialization_steps = 1

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test_using_a_trace_test = False

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
