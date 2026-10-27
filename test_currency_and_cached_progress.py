import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import stock_search as ss


class CurrencyTests(unittest.TestCase):
    def test_indian_listings_are_rupees_even_without_metadata(self):
        for ticker in ['IDEA.NS', 'RELIANCE.NS', '500325.BO']:
            for info in [{}, {'currency': 'USD'}]:
                with patch.object(ss.yf, 'Ticker') as provider:
                    provider.return_value.info = info
                    self.assertEqual(ss.get_company_profile(ticker)['currency'], 'INR')
            with patch.object(ss.yf, 'Ticker', side_effect=RuntimeError('Unavailable')):
                self.assertEqual(ss.get_company_profile(ticker)['currency'], 'INR')
        self.assertEqual(ss.currency_symbol('INR'), '₹')

    def test_us_and_other_reported_currencies(self):
        self.assertEqual(ss.quote_currency('MSFT'), 'USD')
        self.assertEqual(ss.quote_currency('VOD', 'USD'), 'USD')
        self.assertEqual(ss.quote_currency('VOD.L', 'GBp'), 'GBp')
        self.assertEqual(ss.quote_currency('SHOP.TO', 'CAD'), 'CAD')


class CachedProgressTests(unittest.TestCase):
    def test_real_cache_wrapper_renders_and_replays_progress(self):
        from streamlit.testing.v1 import AppTest
        source = Path(__file__).with_name('dashboard.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'cached_engine')
        function = '\n'.join(source.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno])
        script = '''
import time
import streamlit as st
from types import SimpleNamespace
import pandas as pd
import ui_theme as ui
dl = SimpleNamespace(MODEL_VERSION='cache-progress-test')
def cached_load_data(*args):
    return pd.DataFrame({'Close': [1.]})
class Engine:
    def __init__(self, *args, on_progress=None, **kwargs):
        self.on_progress = on_progress
        self.on_progress('ARIMA history: 250/1000 sessions')
    def fit(self):
        self.on_progress('Fwd_Ret_1: validating base features')
        self.on_progress('Validating next-session opening price')
        return self
fe = SimpleNamespace(ForecastEngine=Engine, HORIZONS=[1,5,10,21,63])
'''+function+'''
engine = cached_engine('IDEA.NS', '2020-01-01', '2026-01-01', 'test')
st.success('Training completed')
'''
        app = AppTest.from_string(script, default_timeout=30)
        for _ in range(2):
            app.run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.success[0].value, 'Training completed')


if __name__ == '__main__':
    unittest.main()
