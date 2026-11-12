"""Synthetic checks for the participant-level timeline parser."""
from __future__ import annotations

import sys

import numpy as np

from config import ROOT
from parse_participant_timelines import parse_participants

sys.path.insert(0, str(ROOT / "minute_timeline" / "code"))
import parse_minutes  # noqa: E402


def fixture(cutoff_frame: int = 600239, match_id: str = "TEST") -> dict:
    parts = [{"participantId": i, "puuid": f"person{i}"} for i in range(1, 11)]
    events = [
        {"type": "WARD_PLACED", "timestamp": t, "creatorId": 1, "wardType": w}
        for t, w in [(0, "YELLOW_TRINKET"), (60000, "CONTROL_WARD"), (60001, "SIGHT_WARD"),
                     (600239, "BLUE_TRINKET"), (600240, "SIGHT_WARD"), (1, "UNDEFINED")]
    ]
    events += [
        {"type": "ITEM_PURCHASED", "timestamp": 120000, "participantId": 6, "itemId": 2055},
        {"type": "ITEM_UNDO", "timestamp": 120001, "participantId": 6, "beforeId": 2055, "afterId": 0},
        {"type": "WARD_KILL", "timestamp": 300000, "killerId": 7, "wardType": "SIGHT_WARD"},
        {"type": "WARD_KILL", "timestamp": 300001, "killerId": 7, "wardType": "UNDEFINED"},
        {"type": "CHAMPION_KILL", "timestamp": 400000, "killerId": 2, "victimId": 8, "assistingParticipantIds": [3, 4]},
        {"type": "CHAMPION_KILL", "timestamp": 700000, "killerId": 2, "victimId": 8, "assistingParticipantIds": [3]},
    ]
    pf = {str(i): {"participantId": i, "position": {"x": 100 * i, "y": 50 * i}, "totalGold": 1000 + i, "xp": 2000 + i,
                   "level": 5, "minionsKilled": 40 + i, "jungleMinionsKilled": i} for i in range(1, 11)}
    stamps = [0, 60000, 120000, 300000, cutoff_frame, 900000]
    frames = [{"timestamp": t, "participantFrames": pf, "events": []} for t in stamps]
    frames[-2]["events"] = events
    return {"metadata": {"matchId": match_id}, "info": {"participants": parts, "frames": frames}}


def main() -> None:
    cutoff, valid, puuids, counts, combat, state = parse_participants(fixture(), "TEST")
    assert valid and cutoff == 600239 and puuids[1] == "person1"
    assert counts[0, 0].tolist() == [2, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert counts[0, 4, 0] == 1 and counts[0, 4].sum() == 1
    assert counts[5, 2].sum() == 1 and counts[5, 3, 1] == 1 and counts[5, 3, 2] == -1 and counts[5, 3].sum() == 0
    assert counts[6, 1, 4] == 1 and counts[6, 1].sum() == 1
    assert combat[1].tolist() == [1, 0, 0] and combat[7].tolist() == [0, 1, 0]
    assert combat[2].tolist() == [0, 0, 1] and combat[3].tolist() == [0, 0, 1]
    assert state[0, 0] == 100 and state[9, 1] == 500 and state[4, 2] == 1005

    team = parse_minutes.minute_counts(fixture(), "TEST", cutoff, fast=False)
    for side, pids in [(0, slice(0, 5)), (1, slice(5, 10))]:
        assert np.array_equal(counts[pids, :4].sum(axis=0), team[side]), "participant sums differ from team parser"

    _, invalid, _, counts_invalid, _, _ = parse_participants(fixture(cutoff_frame=598000), "TEST")
    assert not invalid and counts_invalid.sum() == 0
    try:
        parse_participants(fixture(), "OTHER")
    except ValueError as exc:
        assert str(exc) == "match_id_mismatch"
    else:
        raise AssertionError("mismatched match ID accepted")
    print("participant parser checks passed")


if __name__ == "__main__":
    main()
