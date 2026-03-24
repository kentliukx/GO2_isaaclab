import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg, DCMotorCfg

from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
USD_PATH = (THIS_DIR / "../../../GO2/assets/usd/go2/go2.usd").resolve()

GO2_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            retain_accelerations=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "FL_thigh_joint": 0.8, "FL_hip_joint": 0.3, "FL_calf_joint": -1.6,
            "FR_thigh_joint": 0.8, "FR_hip_joint": -0.3, "FR_calf_joint": -1.6,
            "RL_thigh_joint": 0.8, "RL_hip_joint": 0.3, "RL_calf_joint": -1.6,
            "RR_thigh_joint": 0.8, "RR_hip_joint": -0.3, "RR_calf_joint": -1.6
        },
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "hip_and_thigh_actuator": DCMotorCfg(
            joint_names_expr=[
                "FL_thigh_joint", "FL_hip_joint",
                "FR_thigh_joint", "FR_hip_joint",
                "RL_thigh_joint", "RL_hip_joint",
                "RR_thigh_joint", "RR_hip_joint",
            ],
            effort_limit=23.7,
            saturation_effort=23.7,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "calf_actuator": DCMotorCfg(
            joint_names_expr=[
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
            ],
            effort_limit=35.5,
            saturation_effort=35.5,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        )
    },
    actuator_value_resolution_debug_print=True,
)