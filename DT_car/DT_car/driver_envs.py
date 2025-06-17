# driver_envs.py

from highway_env.envs.highway_env import HighwayEnv
from gymnasium.envs.registration import register


class AggressiveHighwayEnv(HighwayEnv):
    def _reward(self, action):
        reward = 0.1 * self.vehicle.speed
        if self.vehicle.lane_index[2] != self.vehicle.target_lane_index[2]:
            reward += 0.2  # more reward for lane change
        if self.vehicle.crashed:
            reward -= 1.0
        return reward


class CautiousHighwayEnv(HighwayEnv):
    def _reward(self, action):
        reward = 0.05 * (30 - self.vehicle.speed)  # slower speed preferred
        if self.vehicle.crashed:
            reward -= 3.0
        if abs(self.vehicle.lane_index[2] - self.vehicle.target_lane_index[2]) > 0:
            reward -= 0.2  # penalize lane change
        return reward


def register_custom_envs():
    register(id='highway-aggressive-v0', entry_point='driver_envs:AggressiveHighwayEnv')
    register(id='highway-cautious-v0', entry_point='driver_envs:CautiousHighwayEnv')
