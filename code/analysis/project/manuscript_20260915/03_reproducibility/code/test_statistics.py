from settings import *
from statistics_core import *
from run_prediction import auc_influence
from run_models import regional_profiles
import unittest
from sklearn.metrics import roc_auc_score


class StatisticalTests(unittest.TestCase):
    def test_auc_placements_with_ties(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.2, 0.3, 0.5, 0.3, 0.5, 0.8])
        a, psi = auc_influence(y, p)
        self.assertAlmostEqual(a, roc_auc_score(y, p), places=12)
        self.assertAlmostEqual(psi.sum(), 0, places=12)

    def test_scale_is_training_only(self):
        r = np.random.default_rng(99)
        train = pd.DataFrame(
            {
                s + "_" + c: r.poisson(5, 1000)
                for s in ["blue", "red"]
                for c in COMPONENTS
            }
        )
        scale = MaintenanceScale().fit(train)
        before = json.dumps(scale.metadata(), sort_keys=True)
        scale.apply(train * 100000)
        self.assertEqual(before, json.dumps(scale.metadata(), sort_keys=True))

    def test_common_region_scale(self):
        r = np.random.default_rng(99)
        train = pd.DataFrame(
            {
                s + "_" + c: r.poisson(5, 1000)
                for s in ["blue", "red"]
                for c in COMPONENTS
            }
        )
        scale = MaintenanceScale().fit(train)
        allv = scale.apply(train)
        subset = scale.apply(train.iloc[100:200])
        np.testing.assert_allclose(
            allv.maintenance_diff_z.iloc[100:200], subset.maintenance_diff_z
        )

    def test_score_bootstrap_against_refits(self):
        rng = np.random.default_rng(812)
        n = 5000
        df = pd.DataFrame(
            {
                "region": rng.choice(REGIONS, n),
                "tier": rng.choice(TIERS, n),
                "patch": rng.choice(["1", "2"], n),
                "maintenance_diff_z": rng.normal(size=n),
            }
        )
        for c in STATE:
            df[c] = rng.normal(size=n)
        df["first_post_neutral_blue"] = rng.binomial(
            1, special.expit(0.15 + 0.3 * df.maintenance_diff_z + 0.2 * df[STATE[0]])
        )
        f = binary_fit(df)
        b = f["effects"][0]["beta"]
        errors = []
        for i in range(30):
            ids = rng.integers(0, n, n)
            refit = binary_fit(df.iloc[ids])
            approx = b + f["psi"][ids, 0].sum() - f["psi"][:, 0].sum()
            errors.append(abs(refit["effects"][0]["beta"] - approx))
        write_json(
            QA / "one_step_refit_validation.json",
            {
                "n": n,
                "resamples": 30,
                "max_absolute_beta_error": max(errors),
                "mean_absolute_beta_error": float(np.mean(errors)),
                "tolerance": 0.02,
            },
        )
        self.assertLess(max(errors), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
