"""Numerical and causality checks for incremental ARIMA filtering."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import forecast_engine as fe


class IncrementalArimaTests(unittest.TestCase):
    def test_fast_budget_preserves_full_mode(self):
        fast = fe._full_members(fast=True)
        full = fe._full_members()
        self.assertEqual(fast['Random Forest'].n_estimators, 60)
        self.assertEqual(fast['XGBoost'].n_estimators, 100)
        self.assertEqual(full['Random Forest'].n_estimators, 300)
        self.assertEqual(full['XGBoost'].n_estimators, 300)

    def test_fast_history_is_bounded_without_mutating_source(self):
        raw = pd.DataFrame({'Close': np.arange(1500) + 100.}, index=pd.bdate_range('2020-01-01', periods=1500))
        with patch.object(fe, 'build_feature_table', return_value=(pd.DataFrame(), {})) as build, \
             patch('volatility_model.fit_horizon_vols', return_value={}):
            engine = fe.ForecastEngine(raw, 'TEST', fast=True)
        self.assertEqual(len(engine.raw), 1000)
        self.assertEqual(len(raw), 1500)
        self.assertEqual(build.call_args.args[0].index.min(), raw.index[500])

    def test_matches_full_history_filter_across_refits(self):
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(size=390))
        raw = pd.DataFrame({'Close': close}, index=pd.bdate_range('2024-01-01', periods=len(close)))
        for order in [(2, 1, 0), (1, 1, 1)]:
            with patch('model_arima.select_best_order', return_value=order):
                actual = fe._arima_features(raw, [1, 5])
            expected = np.full((len(close), 2), np.nan)
            for i in range(fe._ARIMA_WARMUP, len(close)):
                if (i - fe._ARIMA_WARMUP) % fe._ARIMA_REFIT == 0:
                    fitted = ARIMA(close[:i], order=order).fit()
                else:
                    fitted = fitted.apply(close[:i])
                path = fitted.forecast(5)
                expected[i] = path[[0, 4]] / close[i - 1] - 1
            np.testing.assert_allclose(actual.values, expected, rtol=1e-7, atol=1e-9, equal_nan=True)

    def test_future_prices_do_not_change_earlier_features(self):
        raw = pd.DataFrame({'Close': 100 + np.cumsum(np.random.default_rng(3).normal(size=340))})
        changed = raw.copy()
        changed.loc[310:, 'Close'] *= 2
        with patch('model_arima.select_best_order', return_value=(2, 1, 0)):
            before = fe._arima_features(raw, [1, 5])
            after = fe._arima_features(changed, [1, 5])
        np.testing.assert_allclose(before.iloc[:311], after.iloc[:311], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
