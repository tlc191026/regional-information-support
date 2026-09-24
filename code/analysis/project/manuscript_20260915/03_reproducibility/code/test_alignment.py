from settings import *
import unittest
from test_parser import fixture
from parse_aligned import aligned_parser


class AlignmentTests(unittest.TestCase):
    def test_exposure_and_outcome_share_actual_frame_cutoff(self):
        f = fixture(
            [
                {
                    "type": "WARD_PLACED",
                    "timestamp": 600100,
                    "creatorId": 1,
                    "wardType": "CONTROL_WARD",
                },
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 600100,
                    "killerTeamId": 100,
                    "monsterType": "DRAGON",
                },
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": 650000,
                    "killerTeamId": 200,
                    "monsterType": "DRAGON",
                },
            ]
        )
        f["timeline"]["info"]["frames"][10]["timestamp"] = 600250
        rows, _, _ = aligned_parser(f, "TEST")
        m = {r["window"]: r for r in rows}
        self.assertEqual(m["0_10"]["analysis_cutoff_ms"], 600250)
        self.assertEqual(m["0_10"]["state_lag_ms"], 0)
        self.assertEqual(m["0_10"]["blue_valid_wards_placed"], 1)
        self.assertEqual(m["0_10"]["first_post_neutral_blue"], 0)
        self.assertEqual(m["strict_0_10"]["blue_valid_wards_placed"], 0)
        self.assertEqual(m["strict_0_10"]["first_post_neutral_blue"], 1)
        self.assertEqual(m["strict_0_10"]["state_frame_ms"], 540000)

    def test_missing_nearby_frame_excluded(self):
        f = fixture()
        f["timeline"]["info"]["frames"][10]["timestamp"] = 610000
        rows, _, _ = aligned_parser(f, "TEST")
        m = {r["window"]: r for r in rows}
        self.assertFalse(m["0_10"]["window_valid"])
        self.assertEqual(
            m["0_10"]["window_reason"], "no_state_frame_within_one_second_of_landmark"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
