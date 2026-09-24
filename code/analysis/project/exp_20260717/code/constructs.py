"""Single source of truth for the 2026-07-17 construct revision.

The taxonomy distinguishes two observable team-process families from two
task-performance benchmarks.  Scores are theory-coded formative indices;
they are not reflective latent variables.
"""

from __future__ import annotations

MASTER_SEED = 191026
RELEASE_ID = "exp-20260717-signal-taxonomy-seed191026"

REGION_ORDER = ["NA1", "EUW1", "KR"]
REGION_LABELS = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}
REGION_COLORS = {"NA1": "#3B6EA8", "EUW1": "#6B8F71", "KR": "#B05A4A"}
REGION_PAIR_COLORS = {
    "KR-NA": "#B05A4A",
    "KR-EUW": "#6B8F71",
    "EUW-NA": "#3B6EA8",
}

TIER_ORDER = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_INTERDEPENDENCE = {
    "TOP": "Lower",
    "MIDDLE": "Medium",
    "BOTTOM": "Medium",
    "JUNGLE": "High",
    "UTILITY": "High",
}
INTERDEPENDENCE_ORDER = ["Lower", "Medium", "High"]

# Two process families and two outcome/performance benchmarks.  The latter are
# deliberately not described as cooperative-process constructs.
PRIMARY_MEASURE_ORDER = [
    "shared_information_maintenance",
    "team_signalling",
    "joint_action_participation",
    "combat_economy_performance",
]
PRIMARY_MEASURE_LABELS = {
    "shared_information_maintenance": "Shared-information maintenance",
    "team_signalling": "Team signalling",
    "joint_action_participation": "Joint-action participation",
    "combat_economy_performance": "Combat-economy performance",
}
PRIMARY_MEASURE_TYPES = {
    "shared_information_maintenance": "process_family",
    "team_signalling": "process_family",
    "joint_action_participation": "task_performance_indicator",
    "combat_economy_performance": "task_performance_benchmark",
}
PRIMARY_MEASURE_COLORS = {
    "shared_information_maintenance": "#236A64",
    "team_signalling": "#5E8FA3",
    "joint_action_participation": "#B88A3B",
    "combat_economy_performance": "#777777",
}

SIGNAL_SUBTYPE_ORDER = [
    "task_state_attention_signals",
    "coordination_request_signals",
    "action_oriented_signals",
]
SIGNAL_SUBTYPE_LABELS = {
    "task_state_attention_signals": "Task-state and attention signals",
    "coordination_request_signals": "Coordination requests",
    "action_oriented_signals": "Action-oriented signals",
}
SIGNAL_SUBTYPE_COLORS = {
    "task_state_attention_signals": "#8CB7C7",
    "coordination_request_signals": "#5E8FA3",
    "action_oriented_signals": "#315F73",
}

SCORE_ORDER = PRIMARY_MEASURE_ORDER + SIGNAL_SUBTYPE_ORDER
SCORE_LABELS = {**PRIMARY_MEASURE_LABELS, **SIGNAL_SUBTYPE_LABELS}
SCORE_COLORS = {**PRIMARY_MEASURE_COLORS, **SIGNAL_SUBTYPE_COLORS}

# The 18 displayed metrics include vision score as a convergent diagnostic.
# Seventeen metrics enter the main analyses. Vision score is not part of any
# formative score because it mechanically summarizes several maintenance acts.
METRICS = [
    dict(
        metric="vision_score_pm",
        label="Vision score",
        short="Vision score",
        family="shared_information_maintenance",
        subtype="vision_score_diagnostic",
        role="convergent_diagnostic",
        reverse=False,
        primary_score=False,
        primary_boundary=False,
    ),
    dict(
        metric="wards_placed_pm",
        label="Wards placed",
        short="Wards placed",
        family="shared_information_maintenance",
        subtype="information_deployment",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="wards_killed_pm",
        label="Wards cleared",
        short="Wards cleared",
        family="shared_information_maintenance",
        subtype="information_denial",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="control_wards_bought_pm",
        label="Control wards purchased",
        short="Control wards",
        family="shared_information_maintenance",
        subtype="resource_investment",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="enemy_missing_pings_pm",
        label="Enemy-missing pings",
        short="Enemy missing",
        family="team_signalling",
        subtype="task_state_attention_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="enemy_vision_pings_pm",
        label="Enemy-vision pings",
        short="Enemy vision",
        family="team_signalling",
        subtype="task_state_attention_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="command_pings_pm",
        label="Command/attention pings",
        short="Command/attention",
        family="team_signalling",
        subtype="task_state_attention_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="need_vision_pings_pm",
        label="Need-vision pings",
        short="Need vision",
        family="team_signalling",
        subtype="coordination_request_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="assist_me_pings_pm",
        label="Assist-me pings",
        short="Assist me",
        family="team_signalling",
        subtype="coordination_request_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="on_my_way_pings_pm",
        label="On-my-way pings",
        short="On my way",
        family="team_signalling",
        subtype="action_oriented_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="push_pings_pm",
        label="Push pings",
        short="Push",
        family="team_signalling",
        subtype="action_oriented_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="all_in_pings_pm",
        label="All-in pings",
        short="All in",
        family="team_signalling",
        subtype="action_oriented_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="get_back_pings_pm",
        label="Get-back pings",
        short="Get back",
        family="team_signalling",
        subtype="action_oriented_signals",
        role="process_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="assists_pm",
        label="Assists",
        short="Assists",
        family="joint_action_participation",
        subtype="joint_action_participation",
        role="task_performance_indicator",
        reverse=False,
        primary_score=True,
        primary_boundary=False,
    ),
    dict(
        metric="kills_pm",
        label="Kills",
        short="Kills",
        family="combat_economy_performance",
        subtype="combat_economy_performance",
        role="task_performance_benchmark",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="deaths_pm",
        label="Fewer deaths",
        short="Fewer deaths",
        family="combat_economy_performance",
        subtype="combat_economy_performance",
        role="task_performance_benchmark",
        reverse=True,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="gold_earned_pm",
        label="Gold earned",
        short="Gold",
        family="combat_economy_performance",
        subtype="combat_economy_performance",
        role="task_performance_benchmark",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
    dict(
        metric="total_damage_dealt_pm",
        label="Damage dealt",
        short="Damage dealt",
        family="combat_economy_performance",
        subtype="combat_economy_performance",
        role="task_performance_benchmark",
        reverse=False,
        primary_score=True,
        primary_boundary=True,
    ),
]

# Compatibility keys keep the analysis code explicit while allowing a single
# metric dictionary to drive long-format models and figures.
for _item in METRICS:
    _item["domain"] = _item["family"]
    _item["subcomponent"] = _item["subtype"]

METRIC_NAMES = [item["metric"] for item in METRICS]
PRIMARY_SCORE_METRICS = [item["metric"] for item in METRICS if item["primary_score"]]
PRIMARY_BOUNDARY_METRICS = [
    item["metric"] for item in METRICS if item["primary_boundary"]
]

SCORE_METRICS = {
    "shared_information_maintenance": [
        "wards_placed_pm",
        "wards_killed_pm",
        "control_wards_bought_pm",
    ],
    "task_state_attention_signals": [
        "enemy_missing_pings_pm",
        "enemy_vision_pings_pm",
        "command_pings_pm",
    ],
    "coordination_request_signals": ["need_vision_pings_pm", "assist_me_pings_pm"],
    "action_oriented_signals": [
        "on_my_way_pings_pm",
        "push_pings_pm",
        "all_in_pings_pm",
        "get_back_pings_pm",
    ],
    "joint_action_participation": ["assists_pm"],
    "combat_economy_performance": [
        "kills_pm",
        "deaths_pm",
        "gold_earned_pm",
        "total_damage_dealt_pm",
    ],
}

# A hierarchical equal-weight score prevents the four action-oriented signals
# from dominating the two request signals simply because that subtype contains
# more recorded ping categories.
HIERARCHICAL_SCORE_COMPONENTS = {
    "team_signalling": SIGNAL_SUBTYPE_ORDER,
}

# Canonical score hierarchy used by every analysis script. The aliases named
# DOMAIN_* and SUBCOMPONENT_* are retained as implementation vocabulary only;
# manuscript-facing reports use "measure", "process family", "signal subtype"
# and "benchmark" according to PRIMARY_MEASURE_TYPES.
SUBCOMPONENT_ORDER = [
    "shared_information_maintenance",
    *SIGNAL_SUBTYPE_ORDER,
    "joint_action_participation",
    "combat_economy_performance",
]
SUBCOMPONENT_METRICS = {name: SCORE_METRICS[name] for name in SUBCOMPONENT_ORDER}
SUBCOMPONENT_LABELS = {
    "shared_information_maintenance": PRIMARY_MEASURE_LABELS[
        "shared_information_maintenance"
    ],
    **SIGNAL_SUBTYPE_LABELS,
    "joint_action_participation": PRIMARY_MEASURE_LABELS["joint_action_participation"],
    "combat_economy_performance": PRIMARY_MEASURE_LABELS["combat_economy_performance"],
    "vision_score_diagnostic": "Vision-score convergent diagnostic",
}
SUBCOMPONENT_COLORS = {
    "shared_information_maintenance": PRIMARY_MEASURE_COLORS[
        "shared_information_maintenance"
    ],
    **SIGNAL_SUBTYPE_COLORS,
    "joint_action_participation": PRIMARY_MEASURE_COLORS["joint_action_participation"],
    "combat_economy_performance": PRIMARY_MEASURE_COLORS["combat_economy_performance"],
    "vision_score_diagnostic": "#A9C8C4",
}
DOMAIN_ORDER = PRIMARY_MEASURE_ORDER
DOMAIN_LABELS = PRIMARY_MEASURE_LABELS
DOMAIN_COLORS = PRIMARY_MEASURE_COLORS
DOMAIN_COMPONENTS = {
    "shared_information_maintenance": ["shared_information_maintenance"],
    "team_signalling": SIGNAL_SUBTYPE_ORDER,
    "joint_action_participation": ["joint_action_participation"],
    "combat_economy_performance": ["combat_economy_performance"],
}

# Boundary permutation is restricted to multi-indicator families. Assists is a
# one-indicator benchmark and cannot define a within-family pair.
BOUNDARY_FAMILY_ORDER = [
    "shared_information_maintenance",
    "team_signalling",
    "combat_economy_performance",
]

# Historical four-channel coding is retained only as a sensitivity analysis.
LEGACY_FOUR_CHANNELS = {
    "information_support": [
        "vision_score_pm",
        "wards_placed_pm",
        "wards_killed_pm",
        "control_wards_bought_pm",
        "enemy_vision_pings_pm",
        "enemy_missing_pings_pm",
        "command_pings_pm",
    ],
    "execution_coordination": [
        "assists_pm",
        "on_my_way_pings_pm",
        "assist_me_pings_pm",
        "push_pings_pm",
    ],
    "risk_reactive_exploratory": [
        "all_in_pings_pm",
        "need_vision_pings_pm",
        "get_back_pings_pm",
    ],
    "combat_economy_performance": [
        "kills_pm",
        "deaths_pm",
        "gold_earned_pm",
        "total_damage_dealt_pm",
    ],
}

EARLY_INFRA_PRIMARY = [
    "valid_wards_placed",
    "valid_ward_kills",
    "control_ward_purchases",
]
EARLY_INFRA_SENSITIVITIES = {
    "primary_three_component": EARLY_INFRA_PRIMARY,
    "minimal_two_component": ["valid_wards_placed", "valid_ward_kills"],
    "deployment_only": ["control_wards_placed", "valid_ward_kills"],
    "disjoint_four_component": [
        "noncontrol_valid_wards_placed",
        "valid_ward_kills",
        "control_ward_purchases",
        "control_wards_placed",
    ],
}
