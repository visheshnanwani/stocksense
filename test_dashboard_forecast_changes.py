"""Offline regression tests for persistent UI results and causal forecasts."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import dashboard_state as ds
import forecast_engine as fe
import ui_theme as ui


class DashboardStateTests(unittest.TestCase):
    def test_results_survive_rerun_but_not_input_changes(self):
        state, compute = {}, Mock(return_value={"price": 100})
        self.assertTrue(ds.requested(state, "prediction", ("MSFT", 5), True))
        first = ds.result(state, "prediction", compute)
        self.assertTrue(ds.requested(state, "prediction", ("MSFT", 5), False))
        self.assertIs(ds.result(state, "prediction", compute), first)
        compute.assert_called_once()
        self.assertFalse(ds.requested(state, "prediction", ("MSFT", 21), False))
        self.assertNotIn("prediction", state)

    def test_explicit_refresh_recomputes(self):
        state, compute = {}, Mock(side_effect=[1, 2])
        for expected in (1, 2):
            ds.requested(state, "scan", ("BTC",), True)
            self.assertEqual(ds.result(state, "scan", compute), expected)

    def test_unknown_market_state_does_not_claim_closed(self):
        with patch.object(ui, "render") as render:
            ui.hero("Example", "EX", [], 100, 1, 1, None, "22 Oct 10:00", "$")
        markup = render.call_args.args[0]
        self.assertIn("Latest candle", markup)
        self.assertNotIn("Market closed", markup)
        self.assertNotIn("live quote", markup)


class ForecastValidationTests(unittest.TestCase):
    def test_horizon_labels_do_not_cross_validation_boundary(self):
        frame = pd.DataFrame({"feature": np.arange(600)})
        for horizon in (1, 5, 21, 63):
            for train, val in fe._folds(frame, gap=horizon):
                self.assertLess(train.index[-1] + horizon, val.index[0])
                self.assertGreater(len(train), 0)

    def test_cv_passes_target_horizon_to_folds(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"x": rng.normal(size=500), "Fwd_Ret_21": rng.normal(size=500)})
        with patch.object(fe, "_folds", wraps=fe._folds) as folds:
            score = fe._cv_score(frame, ["x"], "Fwd_Ret_21", lambda: [Ridge()])
        self.assertEqual(folds.call_args.kwargs["gap"], 21)
        self.assertTrue(np.isfinite(score["rmse"]))

    def test_historical_prediction_refits_without_future_rows(self):
        idx = pd.bdate_range("2020-01-01", periods=500)
        raw = pd.DataFrame({"Close": np.linspace(100, 150, 500), "Open": np.linspace(99,149,500)}, index=idx)
        table = pd.DataFrame({"x": 1.0, "Fwd_Ret_1": .01, "Fwd_Open_1": .005}, index=idx)
        engine = fe.ForecastEngine.__new__(fe.ForecastEngine)
        engine.raw, engine.table, engine.ticker = raw, table, "EXAMPLE"
        engine.intraday, engine.feature_inputs = False, {}
        engine.models = {1: Mock(predict=Mock(side_effect=AssertionError("future-trained model used")))}
        model = Mock(cols=["x"], groups_used=["base"])
        model.fit_final.return_value = model
        model.predict.return_value = np.array([.01])
        model.band_offsets.return_value = (-.02, .03)
        stats = dict(weights={"Ridge": 1}, alpha=1, sigma=.01, rmse=.01, naive_rmse=.02)
        target, prev = idx[400], idx[399]
        with patch.object(fe, "build_feature_table", return_value=(table.loc[:prev], {"base":["x"]})) as build, \
             patch.object(fe, "_train_horizon", return_value=model) as train, \
             patch.object(fe, "_cv_score", return_value=stats), \
             patch.object(fe.HorizonModel, "fit_final", return_value=model):
            result = engine.predict_date(target)
        self.assertEqual(build.call_args.args[0].index.max(), prev)
        self.assertLess(train.call_args.args[0].index.max(), prev)
        self.assertEqual(result["reference_date"], prev)
        self.assertAlmostEqual(result["predicted_price"], raw.loc[prev, "Close"] * 1.01)
        engine.models[1].predict.assert_not_called()

    def test_evaluation_uses_production_model_and_reports_coverage(self):
        idx = pd.bdate_range("2020-01-01", periods=500)
        raw = pd.DataFrame({"Close": np.linspace(100, 150, 500)}, index=idx)
        table = pd.DataFrame({"x": 1., "Fwd_Ret_1": .01}, index=idx)
        model = Mock(groups_used=["base"])
        model.fit_final.return_value = model
        model.predict.side_effect = lambda rows: np.full(len(rows), .01)
        model.band_offsets.return_value = (-.02, .02)
        with patch.object(fe, "build_feature_table", return_value=(table, {"base":["x"]})), \
             patch.object(fe, "_train_horizon", return_value=model) as train:
            result = fe.evaluate_engine(raw, "EXAMPLE", horizons=[1])
        self.assertEqual(train.call_count, 2)
        self.assertEqual(result.iloc[0]["Price MAE"], 0)
        self.assertEqual(result.iloc[0]["80% interval coverage (%)"], 100)
        # The final training label ends strictly before the held-out boundary.
        self.assertLess(train.call_args.args[0].index.max(), idx[424])


if __name__ == "__main__":
    unittest.main()
