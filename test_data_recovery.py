import unittest
from unittest.mock import patch

import pandas as pd
from yfinance.exceptions import YFRateLimitError
from data_loader import load_stock_data


class DataRecoveryTests(unittest.TestCase):
    @patch('data_loader.time.sleep')
    @patch('data_loader.yf.Ticker')
    @patch('data_loader.yf.download', return_value=pd.DataFrame())
    def test_empty_download_recovers_with_inclusive_dates(self, download, ticker, sleep):
        ticker.return_value.history.return_value = pd.DataFrame(
            {'Close': [100.]}, index=pd.date_range('2026-10-23', periods=1, tz='America/New_York'))
        result = load_stock_data('MSFT', '2026-01-01', '2026-10-23')
        self.assertEqual(result.iloc[0]['Close'], 100.)
        self.assertIsNone(result.index.tz)
        self.assertEqual(ticker.return_value.history.call_args.kwargs['end'], '2026-10-24')
        self.assertEqual(result.attrs['ticker'], 'MSFT')

    @patch('data_loader.time.sleep')
    @patch('data_loader.yf.Ticker')
    @patch('data_loader.yf.download', return_value=pd.DataFrame())
    def test_rate_limit_stops_without_trying_other_symbols(self, download, ticker, sleep):
        ticker.return_value.history.side_effect = YFRateLimitError()
        with self.assertRaisesRegex(ValueError, 'temporarily limiting requests'):
            load_stock_data('TEST.NS', '2026-01-01', '2026-10-23')
        self.assertEqual(download.call_count, 1)
        self.assertEqual(ticker.return_value.history.call_count, 1)
