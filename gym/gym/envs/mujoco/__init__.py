from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv

from gym.envs.mujoco.ant_bandits import AntBanditsEnv
from gym.envs.mujoco.obstacles import Obstacles

from gym.envs.mujoco.ant_movement import AntMovementEnv
from gym.envs.mujoco.ant_obstacles import AntObstaclesEnv
from gym.envs.mujoco.ant_obstaclesbig import AntObstaclesBigEnv
from gym.envs.mujoco.ant_obstaclesgen import AntObstaclesGenEnv

### New one from Andrew
# Primitives for Ant-v1
from gym.envs.mujoco.ant_forward import AntForwardEnv
from gym.envs.mujoco.ant_backward import AntBackwardEnv
from gym.envs.mujoco.ant_curb  import AntCurbEnv
# Primitives for Humanoid-v1
from gym.envs.mujoco.humanoid_forward import HumanoidForwardEnv
from gym.envs.mujoco.humanoid_backward import HumanoidBackwardEnv
from gym.envs.mujoco.humanoid_standup import HumanoidStandupEnv
from gym.envs.mujoco.humanoid_crawl import HumanoidCrawlEnv
from gym.envs.mujoco.humanoid_swim_forward import HumanoidSwimForwardEnv
from gym.envs.mujoco.humanoid_swim_backward import HumanoidSwimBackwardEnv
# Primitives for Walker2d
from gym.envs.mujoco.walker2d_forward import Walker2dForwardEnv
from gym.envs.mujoco.walker2d_backward import Walker2dBackwardEnv
from gym.envs.mujoco.walker2d_balance import Walker2dBalanceEnv
from gym.envs.mujoco.walker2d_walk_lower import Walker2dLowerEnv
from gym.envs.mujoco.walker2d_standup import Walker2dStandupEnv
from gym.envs.mujoco.walker2d_jump import Walker2dJumpEnv
# Meta task for Ant
from gym.envs.mujoco.ant_back_and_forth import AntBackForthEnv
# Meta task for Humanoid
from gym.envs.mujoco.meta_humanoid_v1 import MetaHumanoid_v1_Env
# Meta task for Walker2d
from gym.envs.mujoco.meta_walker2d_v1 import MetaWalker2d_v1_Env
from gym.envs.mujoco.meta_walker2d_v2 import MetaWalker2d_v2_Env
from gym.envs.mujoco.meta_walker2d_v3 import MetaWalker2d_v3_Env

# Primitives for Jaco
from gym.envs.mujoco.jaco_pick import JacoPickEnv
from gym.envs.mujoco.jaco_pick_smaller_box import JacoPickSmallerBoxEnv
from gym.envs.mujoco.jaco_catch import JacoCatchEnv
from gym.envs.mujoco.jaco_shoot import JacoShootEnv
from gym.envs.mujoco.jaco_place import JacoPlaceEnv
from gym.envs.mujoco.jaco_place_only import JacoPlaceOnlyEnv
# Meta task for Jaco
from gym.envs.mujoco.meta_jaco_v1 import MetaJaco_v1_Env
from gym.envs.mujoco.meta_jaco_v2 import MetaJaco_v2_Env

# New humanoid
from gym.envs.mujoco.humanoid_jump import HumanoidJumpEnv
from gym.envs.mujoco.walker2d_backNforth import Walker2dBackNForthEnv
from gym.envs.mujoco.humanoid_hurdle import HumanoidHurdleEnv
from gym.envs.mujoco.humanoid_moving_floor import HumanoidMovingFloorEnv
