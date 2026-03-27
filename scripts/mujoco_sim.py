from __future__ import annotations

import argparse
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "scripts" / "rsl_rl" / "Training_log" / "model_20260326_1.pt"
DEFAULT_URDF_PATH = REPO_ROOT / "source" / "GO2" / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"

DEFAULT_JOINT_POS_BY_NAME = {
    "FL_hip_joint": 0.3,
    "FR_hip_joint": -0.3,
    "RL_hip_joint": 0.3,
    "RR_hip_joint": -0.3,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 0.8,
    "RR_thigh_joint": 0.8,
    "FL_calf_joint": -1.6,
    "FR_calf_joint": -1.6,
    "RL_calf_joint": -1.6,
    "RR_calf_joint": -1.6,
}
ISAAC_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
COMMAND_LIMITS = np.array([[ -2.0, 2.0], [ -1.0, 1.0], [ -2.0, 2.0]], dtype=np.float64)
OBS_DIM = 45
ACTION_DIM = len(DEFAULT_JOINT_POS_BY_NAME)
GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float64)
RIGID_CONTACT_SOLREF = np.array([0.002, 1.0], dtype=np.float64)
RIGID_CONTACT_SOLIMP = np.array([0.99, 0.999, 0.0001, 0.5, 2.0], dtype=np.float64)


@dataclass
class CommandState:
    x: float
    y: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=np.float32)

    def clamp_(self) -> None:
        clipped = np.clip(self.as_array(), COMMAND_LIMITS[:, 0], COMMAND_LIMITS[:, 1])
        self.x, self.y, self.yaw = map(float, clipped)


class PolicyWrapper(nn.Module):
    def __init__(self, actor: nn.Module, normalizer: nn.Module | None = None):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer if normalizer is not None else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


def build_actor_from_checkpoint(model_state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
    layer_ids = sorted(
        int(key.split(".")[1])
        for key in model_state_dict
        if key.startswith("actor.") and key.endswith(".weight")
    )
    if not layer_ids:
        raise ValueError("Checkpoint does not contain actor weights.")

    modules: list[nn.Module] = []
    for i, layer_id in enumerate(layer_ids):
        weight = model_state_dict[f"actor.{layer_id}.weight"]
        bias = model_state_dict[f"actor.{layer_id}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        modules.append(linear)
        if i != len(layer_ids) - 1:
            modules.append(nn.ELU())
    return nn.Sequential(*modules)


def load_policy(model_path: Path, device: torch.device) -> nn.Module:
    try:
        policy = torch.jit.load(str(model_path), map_location=device)
        policy.eval()
        return policy
    except RuntimeError:
        pass

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported model file. Expected either a TorchScript policy or an RSL-RL checkpoint "
            "with 'model_state_dict'."
        )

    actor = build_actor_from_checkpoint(checkpoint["model_state_dict"])
    normalizer: nn.Module | None = None

    if "obs_norm_state_dict" in checkpoint:
        try:
            from rsl_rl.modules import EmpiricalNormalization
        except ImportError as exc:
            raise ImportError(
                "Checkpoint contains observation normalization state, but rsl_rl is not available."
            ) from exc
        normalizer = EmpiricalNormalization(shape=[OBS_DIM], until=1.0e8)
        normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        normalizer.eval()

    policy = PolicyWrapper(actor=actor, normalizer=normalizer).to(device)
    policy.eval()
    return policy


def build_mujoco_model(
    urdf_path: Path,
    sim_dt: float,
    kp: float,
    kd: float,
    ground_size: float,
) -> tuple[mujoco.MjModel, list[str], dict[str, int], np.ndarray]:
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.option.timestep = sim_dt
    spec.option.gravity = [0.0, 0.0, -9.81]

    base_body = next(body for body in spec.bodies if body.name == "base")
    base_body.add_freejoint()

    ground = spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[ground_size, ground_size, 0.1],
        rgba=[0.2, 0.25, 0.3, 1.0],
    )
    ground.friction = [1.0, 0.005, 0.0001]
    ground.solref = RIGID_CONTACT_SOLREF.copy()
    ground.solimp = RIGID_CONTACT_SOLIMP.copy()

    for body in spec.bodies:
        for geom in body.geoms:
            geom.solref = RIGID_CONTACT_SOLREF.copy()
            geom.solimp = RIGID_CONTACT_SOLIMP.copy()

    spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0], diffuse=[0.9, 0.9, 0.9])

    joint_specs = {joint.name: joint for joint in spec.joints}
    for joint_name in DEFAULT_JOINT_POS_BY_NAME:
        joint = joint_specs[joint_name]
        actuator = spec.add_actuator()
        actuator.name = f"{joint_name}_pos"
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = joint_name
        actuator.set_to_position(kp, kv=kd)
        actuator.ctrlrange = joint.range.copy()
        actuator.ctrllimited = True
        effort_limit = 35.5 if "calf" in joint_name else 23.7
        actuator.forcerange = [-effort_limit, effort_limit]
        actuator.forcelimited = True

    model = spec.compile()
    model_joint_names = {model.joint(i).name for i in range(model.njnt)}
    joint_names = [name for name in ISAAC_JOINT_ORDER if name in model_joint_names]
    if len(joint_names) != ACTION_DIM:
        raise RuntimeError(
            f"Expected {ACTION_DIM} actuated joints, but resolved {len(joint_names)} joints: {joint_names}"
        )
    joint_qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in joint_names}
    default_joint_pos = np.array([DEFAULT_JOINT_POS_BY_NAME[name] for name in joint_names], dtype=np.float64)
    return model, joint_names, joint_qpos_adr, default_joint_pos


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    joint_qpos_adr: dict[str, int],
    default_joint_pos: np.ndarray,
    start_height: float,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0.0, 0.0, start_height], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[:] = 0.0
    for joint_name, default_pos in zip(joint_names, default_joint_pos, strict=True):
        data.qpos[joint_qpos_adr[joint_name]] = default_pos
    data.ctrl[:] = default_joint_pos
    mujoco.mj_forward(model, data)


def compute_projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    inv_quat = base_quat_wxyz.copy()
    inv_quat[1:] *= -1.0
    projected = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(projected, GRAVITY_VEC, inv_quat)
    return projected


def build_observation(
    data: mujoco.MjData,
    joint_names: list[str],
    joint_qpos_adr: dict[str, int],
    default_joint_pos: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    joint_pos_rel = np.array(
        [data.qpos[joint_qpos_adr[name]] for name in joint_names],
        dtype=np.float32,
    ) - default_joint_pos.astype(np.float32)
    joint_vel = np.array(
        [data.joint(name).qvel[0] for name in joint_names],
        dtype=np.float32,
    )
    projected_gravity = compute_projected_gravity(data.qpos[3:7].copy()).astype(np.float32)
    base_ang_vel = np.asarray(data.qvel[3:6], dtype=np.float32)
    obs = np.concatenate(
        [
            command.astype(np.float32),
            last_action.astype(np.float32),
            joint_pos_rel,
            0.05 * joint_vel,
            projected_gravity,
            0.2 * base_ang_vel,
        ]
    )
    if obs.shape[0] != OBS_DIM:
        raise RuntimeError(f"Observation dim mismatch: got {obs.shape[0]}, expected {OBS_DIM}.")
    return obs


def run_policy(policy: nn.Module, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        action = policy(obs_tensor)
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Policy returned unsupported type: {type(action)}")
    return action.squeeze(0).detach().cpu().numpy().astype(np.float64)


def sample_command(rng: np.random.Generator) -> CommandState:
    cmd = rng.uniform(COMMAND_LIMITS[:, 0], COMMAND_LIMITS[:, 1])
    return CommandState(x=float(cmd[0]), y=float(cmd[1]), yaw=float(cmd[2]))


def maybe_sleep(target_dt: float, start_time: float, enable_real_time: bool) -> None:
    if not enable_real_time:
        return
    elapsed = time.time() - start_time
    remaining = target_dt - elapsed
    if remaining > 0.0:
        time.sleep(remaining)


def format_array(name: str, values: np.ndarray) -> str:
    return f"{name}={np.array2string(values, precision=4, floatmode='fixed', suppress_small=False)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GO2 locomotion policy in MuJoCo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to model.pt or policy.pt.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH, help="Path to the GO2 URDF.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo simulation dt.")
    parser.add_argument("--control-decimation", type=int, default=4, help="Policy is evaluated every N sim steps.")
    parser.add_argument(
        "--command-resample-seconds",
        type=float,
        default=10.0,
        help="Resample a random velocity command every N simulated seconds.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for command sampling.")
    parser.add_argument("--kp", type=float, default=25.0, help="Position actuator stiffness.")
    parser.add_argument("--kd", type=float, default=0.5, help="Position actuator damping.")
    parser.add_argument("--start-height", type=float, default=0.3, help="Initial base height.")
    parser.add_argument("--ground-size", type=float, default=5.0, help="Half-size of the ground plane.")
    parser.add_argument("--duration", type=float, default=None, help="Optional max simulated seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer.")
    parser.add_argument(
        "--disable-real-time",
        action="store_true",
        help="Run as fast as possible instead of matching wall-clock time.",
    )
    parser.add_argument(
        "--manual-step",
        action="store_true",
        help="Wait for Enter before each control step and print obs/action for that step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = args.model.resolve()
    args.urdf = args.urdf.resolve()
    device = torch.device(args.device)

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.urdf.exists():
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")

    policy = load_policy(args.model, device)
    model, joint_names, joint_qpos_adr, default_joint_pos = build_mujoco_model(
        urdf_path=args.urdf,
        sim_dt=args.sim_dt,
        kp=args.kp,
        kd=args.kd,
        ground_size=args.ground_size,
    )
    data = mujoco.MjData(model)

    rng = np.random.default_rng(args.seed)
    command_state = sample_command(rng)
    next_command_time = args.command_resample_seconds
    last_action = np.zeros(len(joint_names), dtype=np.float32)
    control_dt = args.sim_dt * args.control_decimation

    reset_state(
        model,
        data,
        joint_names,
        joint_qpos_adr,
        default_joint_pos,
        start_height=args.start_height,
    )

    print(f"[INFO] Loaded policy from: {args.model}")
    print(f"[INFO] Loaded URDF from: {args.urdf}")
    print(f"[INFO] Joint order: {joint_names}")
    print(
        "[INFO] Random command sampler: "
        f"every {args.command_resample_seconds:.1f}s sample vx, vy, yaw from training ranges."
    )
    print(
        "[INFO] Initial command: "
        f"vx={command_state.x:+.2f}, vy={command_state.y:+.2f}, yaw={command_state.yaw:+.2f}"
    )
    if args.manual_step:
        print("[INFO] Manual step mode: press Enter to step, input 'q' then Enter to quit.")

    def step_once() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        nonlocal last_action
        time_before = float(data.time)
        obs = build_observation(
            data=data,
            joint_names=joint_names,
            joint_qpos_adr=joint_qpos_adr,
            default_joint_pos=default_joint_pos,
            last_action=last_action,
            command=command_state.as_array(),
        )
        action = run_policy(policy, obs, device)
        # action = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        target = default_joint_pos + action
        data.ctrl[:] = target
        last_action = action.astype(np.float32)
        mujoco.mj_step(model, data, nstep=args.control_decimation)
        next_obs = build_observation(
            data=data,
            joint_names=joint_names,
            joint_qpos_adr=joint_qpos_adr,
            default_joint_pos=default_joint_pos,
            last_action=last_action,
            command=command_state.as_array(),
        )
        time_after = float(data.time)
        return obs, action, target, next_obs, time_before, time_after

    def maybe_resample_command() -> None:
        nonlocal command_state, next_command_time
        if data.time < next_command_time:
            return
        command_state = sample_command(rng)
        next_command_time += args.command_resample_seconds
        print(
            "[INFO] Resampled command: "
            f"vx={command_state.x:+.2f}, vy={command_state.y:+.2f}, yaw={command_state.yaw:+.2f}"
        )

    if args.headless:
        start_wall = time.time()
        last_log_time = -1.0
        while args.duration is None or data.time < args.duration:
            loop_start = time.time()
            if args.manual_step:
                user_input = input("[STEP] Enter=step, q=quit > ").strip().lower()
                if user_input == "q":
                    break
            maybe_resample_command()
            obs, action, target, next_obs, time_before, time_after = step_once()
            if args.manual_step:
                print(f"[STEP] t_before={time_before:6.3f}s t_after={time_after:6.3f}s")
                print(format_array("obs", obs))
                print(format_array("action", action))
                print(format_array("target_q", target))
                print(format_array("next_obs", next_obs))
            if math.floor(data.time) != math.floor(last_log_time):
                base_pos = data.qpos[:3].copy()
                print(
                    f"[INFO] t={data.time:6.2f}s "
                    f"base=({base_pos[0]:+.2f}, {base_pos[1]:+.2f}, {base_pos[2]:+.2f}) "
                    f"cmd=({command_state.x:+.2f}, {command_state.y:+.2f}, {command_state.yaw:+.2f})"
                )
                last_log_time = data.time
            maybe_sleep(control_dt, loop_start, enable_real_time=not args.disable_real_time)
        print(f"[INFO] Finished headless rollout in {time.time() - start_wall:.2f}s wall-clock.")
        return

    mujoco_viewer = importlib.import_module("mujoco.viewer")
    with mujoco_viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -20.0
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.25])
        viewer.sync()

        while viewer.is_running() and (args.duration is None or data.time < args.duration):
            loop_start = time.time()
            if args.manual_step:
                user_input = input("[STEP] Enter=step, q=quit > ").strip().lower()
                if user_input == "q":
                    break
            maybe_resample_command()
            obs, action, target, next_obs, time_before, time_after = step_once()
            if args.manual_step:
                print(f"[STEP] t_before={time_before:6.3f}s t_after={time_after:6.3f}s")
                print(format_array("obs", obs))
                print(format_array("action", action))
                print(format_array("target_q", target))
                print(format_array("next_obs", next_obs))
            with viewer.lock():
                viewer.cam.lookat[:] = data.body("base").xpos.copy()
            viewer.sync()
            maybe_sleep(control_dt, loop_start, enable_real_time=not args.disable_real_time)


if __name__ == "__main__":
    main()
