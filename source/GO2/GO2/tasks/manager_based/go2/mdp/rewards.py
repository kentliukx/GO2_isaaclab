# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action jerk using the squared second-order finite difference."""
    current_action = env.action_manager.action

    if not hasattr(env, "_action_prev"):
        env._action_prev = current_action.clone()
    if not hasattr(env, "_action_prev_prev"):
        env._action_prev_prev = current_action.clone()

    reset_mask = env.episode_length_buf <= 1
    env._action_prev[reset_mask] = current_action[reset_mask]
    env._action_prev_prev[reset_mask] = current_action[reset_mask]

    second_diff = current_action - 2.0 * env._action_prev + env._action_prev_prev
    reward = torch.sum(torch.square(second_diff), dim=1)

    # The first two steps after a reset do not have enough history for a valid second difference.
    reward = reward * (env.episode_length_buf > 2).float()

    env._action_prev_prev[:] = env._action_prev
    env._action_prev[:] = current_action
    return reward

def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Use the provided target height directly for flat terrain
    adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)

def desired_contacts_sum(env:ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward the numbers of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    contacts_sum = contacts.sum(dim=-1)
    return 1.0 * contacts_sum

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Only penalize short swing durations; do not reward excessively long airtime.
    reward = torch.sum(torch.clamp(last_air_time - threshold, max=0.0) * first_contact, dim=1)
    return reward

def feet_air_time_excess(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Apply a constant penalty when any selected foot stays airborne too long."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.any(current_air_time > threshold, dim=1).float()

def energy_consumption(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize energy consumption."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def foot_clearance(
    env: ManagerBasedRLEnv, target_height: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize toe height deviation during swing based on foot xy-speed.

    The toe position is read from the frame transformer target frames so the reward uses the
    offset foot-tip coordinates instead of rigid-body origins.

    The term matches: sum_i (target_height - p_z^i)^2 * ||v_xy^i||.
    """
    foot_transformer: FrameTransformer = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    toe_pos_w = foot_transformer.data.target_pos_w[:, sensor_cfg.body_ids, :]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]

    body_to_toe_w = toe_pos_w - body_pos_w
    toe_lin_vel_w = body_lin_vel_w + torch.cross(body_ang_vel_w, body_to_toe_w, dim=-1)

    foot_height = toe_pos_w[:, :, 2]
    foot_xy_speed = torch.norm(toe_lin_vel_w[:, :, :2], dim=-1)
    return torch.sum(torch.square(target_height - foot_height) * foot_xy_speed, dim=1)

def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_transformer_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    air_time_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize toe xy-speed squared when the last air time is shorter than a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_transformer: FrameTransformer = env.scene.sensors[foot_transformer_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    short_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids] < air_time_threshold
    toe_pos_w = foot_transformer.data.target_pos_w[:, foot_transformer_cfg.body_ids, :]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]

    body_to_toe_w = toe_pos_w - body_pos_w
    toe_lin_vel_w = body_lin_vel_w + torch.cross(body_ang_vel_w, body_to_toe_w, dim=-1)
    toe_xy_speed_sq = torch.sum(torch.square(toe_lin_vel_w[:, :, :2]), dim=-1)
    return torch.sum(short_air_time.float() * toe_xy_speed_sq, dim=1)