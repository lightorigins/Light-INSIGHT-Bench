"""Task config for the published objnav suite (``insight-bench-v1``).

The suite scores object-goal navigation with the NavNuances LR-towards
protocol -- final position within the episode's success radius of the object
centre, with the object inside the camera's horizontal field of view -- so the
scoring config is the NavNuances one
(:mod:`insight_bench.vln_runtime.scoring.navnuances`) and the per-episode
radius comes from the episode record (see
:mod:`insight_bench.vln_runtime.episodes.adapters.objnav`).

The rig is HFOV 120 with a 200 m far clip plane over a 300-step budget, against
four scene families (HM3D, MP3D, InteriorGS and Habitat-GS).

This rig is what the coordinate is defined at going forward, not a replay of
the sets it was assembled from. Those were measured at two different rigs: the
outdoor Gaussian-splat sets at 300 steps with a 200 m far plane, the indoor
sets at 200 steps with a 100 m far plane. The clip change is a no-op indoors --
nothing indoors is more than 100 m away -- but the step budget is not
calibrated across that difference, so numbers produced here are not expected to
reproduce the previously published indoor figures. The suite's rig is declared
where the runtime reads it, and the manifest's comparability notes say the
same thing to anyone reading a leaderboard row.

Scene assets are user-provided for three of the four families and this SDK
never downloads them, so ``assets["scene_root"]`` has no default and
``require_scene_assets`` is on: a scene that did not load renders a flat frame,
and the rollout refuses that instead of scoring an empty room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from insight_bench.vln_runtime.managers import DoneTerm, MeasureTerm, TerminationTermCfg
from insight_bench.vln_runtime.measures import terms as measure_terms
from insight_bench.vln_runtime.scoring import navnuances_terms
from insight_bench.vln_runtime.scoring.term_cfg import ScoreTerm
from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig, SensorConfig, TerrainConfig
from insight_bench.vln_runtime.terminations import terms as termination_terms

TASK_NAME = "insight_bench"


@dataclass
class ObjnavMeasuresCfg:
    """Online measures that follow rollout state, not offline scoring policy."""

    path_length: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=measure_terms.vlnce_path_length)
    )
    # NOT the run result's distance_to_goal, which is the straight line from
    # where the agent stopped to the target object and is what decides success.
    # This one is route-following: distance to the nearest point on the episode's
    # reference path, plus however much of that path is left beyond it. The two
    # disagree for two reasons -- a reference path ends at a viewpoint a metre or
    # two short of the object, and finishing off-route adds the whole un-walked
    # remainder back on. Measured over a full suite they differ by half a metre
    # at the median and by eleven at the tail. They shared the name
    # distance_to_goal until 0.2.0; the name here says which one it is.
    route_distance_remaining_m: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=measure_terms.route_distance_remaining)
    )
    oracle_navigation_error: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=measure_terms.vlnce_oracle_navigation_error)
    )
    oracle_success: MeasureTerm = field(
        default_factory=lambda: MeasureTerm(func=measure_terms.vlnce_oracle_success)
    )


@dataclass
class ObjnavTerminationsCfg:
    """When an episode ends: the policy said stop, it stood still, or time ran out."""

    policy_stop: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(func=termination_terms.policy_stop_requested)
    )
    zero_velocity_stop: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(
            # Stop on the FIRST near-zero velocity command (single step), matching
            # Habitat's velocity_control is_stop_called semantics. A waypoint
            # policy signals stop by emitting its zero-motion cluster; waiting for
            # a sustained duration made the agent overshoot the goal before halting.
            func=termination_terms.velocity_control_below_threshold,
            params={
                "min_abs_lin_speed_mps": 0.01,
                "min_abs_ang_speed_dps": 0.5,
            },
        )
    )
    time_out: TerminationTermCfg = field(
        default_factory=lambda: DoneTerm(func=termination_terms.time_out, time_out=True)
    )


@dataclass
class ObjnavScoresCfg:
    """The scores a run reports, all from the LR-towards success protocol."""

    success: ScoreTerm = field(default_factory=lambda: ScoreTerm(func=navnuances_terms.success))
    spl: ScoreTerm = field(default_factory=lambda: ScoreTerm(func=navnuances_terms.spl))
    distance_to_goal: ScoreTerm = field(
        default_factory=lambda: ScoreTerm(func=navnuances_terms.distance_to_goal)
    )
    path_length: ScoreTerm = field(
        default_factory=lambda: ScoreTerm(func=navnuances_terms.path_length)
    )
    stop_step: ScoreTerm = field(default_factory=lambda: ScoreTerm(func=navnuances_terms.stop_step))
    capability: ScoreTerm = field(
        default_factory=lambda: ScoreTerm(func=navnuances_terms.capability)
    )
    stopped: ScoreTerm = field(default_factory=lambda: ScoreTerm(func=navnuances_terms.stopped))


# The scene root is user-supplied and licence-gated; an empty sentinel makes an
# unconfigured run fail by naming the asset key instead of reaching for a
# machine-specific path.
DEFAULT_SCENE_ROOT = ""


def default_task_config() -> BenchmarkTaskConfig:
    return BenchmarkTaskConfig(
        name=TASK_NAME,
        backend="camera_walk",
        terrain=TerrainConfig(
            kind="usd",
            prim_path="/World/scene",
            usd_path=None,
            env_spacing_m=1.0,
            metadata={"episode_overrides_usd_path": True},
        ),
        sensors=(
            SensorConfig(
                name="rgb",
                kind="pinhole_camera",
                prim_path="/World/Robot/front_camera",
                width=480,
                height=270,
                params={
                    "camera_height_m": 1.0,
                    "hfov_deg": 120.0,
                    "sim_dt_sec": 1.0 / 30.0,
                    "focal_length_cm": 24.0,
                    "focus_distance_cm": 400.0,
                    # 200 m, not the indoor 100 m: outdoor 3DGS scenes reach
                    # roughly +-115 m in world coordinates, and a 100 m far
                    # plane culls the distant half of a park scene.
                    "clipping_range_m": (0.05, 200.0),
                },
            ),
        ),
        measures=ObjnavMeasuresCfg(),
        terminations=ObjnavTerminationsCfg(),
        scoring=ObjnavScoresCfg(),
        assets={
            "dataset": "insight_bench",
            "scene_root": DEFAULT_SCENE_ROOT,
            # A user-provided USD that failed to load renders flat; refuse it.
            "require_scene_assets": True,
        },
        metadata={
            "benchmark_family": "objnav",
            "task_kind": "object_goal_navigation",
            "scene_format": "usd",
            "success_protocol": "navnuances_lr_towards",
            "stop_termination_terms": ("policy_stop", "zero_velocity_stop"),
            "rollout": {
                "num_steps": 300,
                "warmup_steps": 3,
                # Warmup alone captures the first frame in about 75 ms, while
                # material compilation and texture streaming stay busy for 5-10
                # render steps on a mesh scene -- so the two-gate convergence
                # fallback runs after it. Both values are the calibrated ones:
                # the tolerance is 3x the steady-state jitter measured over 55
                # fixed-pose probe runs (median 0.072, max 0.158 mean |dRGB|),
                # and 60 steps is the budget that bounds the wait. Leaving them
                # at the dataclass defaults of 0 disables the gate entirely,
                # which is how this suite's scene mix first went wrong.
                "render_converge_tol": 0.5,
                "render_converge_max_steps": 60,
                "render_settle_steps": 1,
                "control_dt_sec": 0.1,
                "lin_vel_range_mps": (0.0, 2.5),
                "ang_vel_range_dps": (-300.0, 300.0),
                "waypoint_atol": 1e-6,
                "yaw_hybrid_thresh_deg": 1.0,
                "yaw_lookahead_steps": 1,
                "yaw_lookahead_hybrid": True,
                "waypoint_index": None,
                "drop_lateral": True,
                "motion_scale": 1.0,
                "yaw_scale": 1.0,
            },
        },
    )
