from settings import *
import unittest, copy, io
import pandas as pd
from parse_raw import parse_timeline, parse_matchdto, team_of


def fixture(extra=None):
    frames = []
    for t in range(0, 1260000, 60000):
        pf = {
            str(pid): {
                "participantId": pid,
                "totalGold": 500 + t / 1000,
                "xp": t / 100,
                "level": 5,
                "damageStats": {"totalDamageDoneToChampions": t / 1000},
            }
            for pid in range(1, 11)
        }
        frames.append({"timestamp": t, "participantFrames": pf, "events": []})
    frames[-1]["events"] = [
        *(extra or []),
        {"type": "GAME_END", "timestamp": 1200000, "winningTeam": 100},
    ]
    return {
        "match_id": "TEST",
        "timeline": {
            "metadata": {"matchId": "TEST"},
            "info": {
                "participants": [
                    {"participantId": i, "puuid": f"person{i}"} for i in range(1, 11)
                ],
                "frames": frames,
            },
        },
    }


class ParserTests(unittest.TestCase):
    def row(self, events):
        return parse_timeline(fixture(events), "TEST")[0][0]

    def test_identity_mismatch(self):
        with self.assertRaisesRegex(ValueError, "match_id"):
            parse_timeline(fixture(), "WRONG")

    def test_incomplete_participants(self):
        f = fixture()
        f["timeline"]["info"]["participants"].pop()
        with self.assertRaisesRegex(ValueError, "participant"):
            parse_timeline(f, "TEST")

    def test_boundary_and_control_not_double_counted(self):
        r = self.row(
            [
                {
                    "type": "WARD_PLACED",
                    "timestamp": 600000,
                    "creatorId": 1,
                    "wardType": "CONTROL_WARD",
                },
                {
                    "type": "WARD_PLACED",
                    "timestamp": 600001,
                    "creatorId": 1,
                    "wardType": "YELLOW_TRINKET",
                },
            ]
        )
        self.assertEqual(r["blue_valid_wards_placed"], 1)
        self.assertEqual(r["blue_control_wards_placed"], 1)

    def test_undefined_excluded(self):
        r = self.row(
            [
                {
                    "type": "WARD_PLACED",
                    "timestamp": 100,
                    "creatorId": 1,
                    "wardType": "UNDEFINED",
                }
            ]
        )
        self.assertEqual(r["blue_valid_wards_placed"], 0)
        self.assertEqual(r["blue_raw_wards_placed"], 1)

    def test_undo(self):
        r = self.row(
            [
                {
                    "type": "ITEM_PURCHASED",
                    "timestamp": 100,
                    "participantId": 1,
                    "itemId": 2055,
                },
                {
                    "type": "ITEM_UNDO",
                    "timestamp": 200,
                    "participantId": 1,
                    "beforeId": 2055,
                    "afterId": 0,
                },
            ]
        )
        self.assertEqual(r["blue_control_ward_purchases"], 1)
        self.assertEqual(r["blue_control_ward_undo_net"], 0)

    def test_no_objective_not_zero(self):
        r = self.row([])
        self.assertEqual(r["first_status"], "no_event")
        self.assertTrue(pd.isna(r["first_post_neutral_blue"]))

    def test_neutral_boundary(self):
        r = self.row(
            [
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 600000,
                    "killerTeamId": 100,
                    "monsterType": "DRAGON",
                },
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 600001,
                    "killerTeamId": 200,
                    "monsterType": "DRAGON",
                },
            ]
        )
        self.assertEqual(r["blue_neutral_objectives_pre"], 1)
        self.assertEqual(r["first_post_neutral_blue"], 0)

    def test_unassigned_is_missing(self):
        r = self.row(
            [
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 700000,
                    "killerTeamId": 300,
                    "monsterType": "HORDE",
                },
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 800000,
                    "killerTeamId": 100,
                    "monsterType": "DRAGON",
                },
            ]
        )
        self.assertEqual(r["first_status"], "unassigned")
        self.assertTrue(pd.isna(r["first_post_neutral_blue"]))
        self.assertEqual(r["first_legal_post_neutral_blue"], 1)

    def test_tie_is_missing(self):
        r = self.row(
            [
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 700000,
                    "killerTeamId": t,
                    "monsterType": "HORDE",
                }
                for t in [100, 200]
            ]
        )
        self.assertEqual(r["first_status"], "tied_opposing_teams")
        self.assertTrue(pd.isna(r["first_post_neutral_blue"]))

    def test_same_team_simultaneous_not_tie(self):
        r = self.row(
            [
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 700000,
                    "killerTeamId": 100,
                    "monsterType": "HORDE",
                }
            ]
            * 2
        )
        self.assertEqual(r["first_post_neutral_blue"], 1)

    def test_future_frame_never_used(self):
        f = fixture()
        f["timeline"]["info"]["frames"][10]["timestamp"] = 600001
        r = parse_timeline(f, "TEST")[0][0]
        self.assertEqual(r["state_frame_ms"], 540000)

    def test_building_award_reversed(self):
        r = self.row([{"type": "BUILDING_KILL", "timestamp": 700000, "teamId": 200}])
        self.assertEqual(r["blue_building_objectives_post"], 1)

    def test_na_preserved(self):
        df = pd.read_csv(
            io.StringIO("region,value\nNA,1\nEUW,2\nKR,3\n"), keep_default_na=False
        )
        self.assertEqual(df.region.tolist(), ["NA", "EUW", "KR"])

    def test_side_mapping(self):
        self.assertEqual(
            [team_of(i) for i in [0, 1, 5, 6, 10, 11]], [None, 100, 100, 200, 200, None]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
