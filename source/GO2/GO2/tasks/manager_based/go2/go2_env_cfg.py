# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from GO2.robots.GO2 import GO2_CONFIG
from GO2.terrains.rough import ROUGH_TERRAINS_CFG


##
# Scene definition
##


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for a GO2 scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     debug_vis=False,
    # )

    # robot
    robot: ArticulationCfg = GO2_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        debug_vis=True,
        track_air_time=True,
        force_threshold=1,
    )

    extra_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        debug_vis=True,
        track_air_time=True,
        force_threshold=40,
    )

    # frame transformer
    foot_frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*calf",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.22),
                    rot=(1.0, 0.0, 0.0, 0.0),
                )
            )
        ],
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "FL_thigh_joint", "FL_hip_joint", "FL_calf_joint",
            "FR_thigh_joint", "FR_hip_joint", "FR_calf_joint",
            "RL_thigh_joint", "RL_hip_joint", "RL_calf_joint",
            "RR_thigh_joint", "RR_hip_joint", "RR_calf_joint"],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        generated_commands = ObsTerm(
            func=mdp.generated_commands,
            params={
                "command_name": "base_velocity"
            },
            scale=1.0
        )
        last_action = ObsTerm(func=mdp.last_action, scale=1.0)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Constant running reward
    is_alive = RewTerm(func=mdp.is_alive, weight=5)

    # Primary task: follow commands
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.0, 
        params={
            "std": 0.5,
            "command_name": "base_velocity",
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={
            "std": 0.5,
            "command_name": "base_velocity",
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    
    # Shaping task: walk like a real dog
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.3},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=[".*calf"],
            ),
            "foot_transformer_cfg": SceneEntityCfg(
                name="foot_frame_transformer",
                body_names=[".*calf"],
            ),
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=[".*calf"],
            ),
            "air_time_threshold": 0,
        },
    )
    # foot_clearance = RewTerm(
    #     func=mdp.foot_clearance,
    #     weight=-1, 
    #     params={
    #         "target_height": 0.1,
    #         "sensor_cfg": SceneEntityCfg(
    #             name="foot_frame_transformer",
    #             body_names=[".*calf"],
    #             ),
    #         "asset_cfg": SceneEntityCfg(
    #             name="robot",
    #             body_names=[".*calf"],
    #             ),
    #     },
    # )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=[".*calf"],
            ),
            "threshold": 0.3,
            "max_reward_time": 0.6,
        },
    )
    feet_ground_time = RewTerm(
        func=mdp.feet_ground_time,
        weight=1,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=[".*calf"],
            ),
            "threshold": 0.5
        },
    )
    feet_air_time_excess = RewTerm(
        func=mdp.feet_air_time_excess,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="extra_contact_forces",
                body_names=[".*calf"],
            ),
            "threshold": 0.75,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces", 
                body_names=[".*thigh"]
            ),
            "threshold": 1.0,
        },
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5)

    # Shaping task: slow movements
    energy_consumption = RewTerm(func=mdp.energy_consumption, weight=-1e-4)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-8)
    joint_speed_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) lie down
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*hip"]), "threshold": 1.0},
    )
    # (3) flip over
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 2},
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1, 1), lin_vel_y=(-1, 1), ang_vel_z=(-1, 1)
        )
    )

##
# Environment configuration
##


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
