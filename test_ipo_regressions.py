"""Offline regression checks for IPO analysis and source parsing."""
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import ipo_analyzer as ia


class IPORegressionTests(unittest.TestCase):
    def test_analysis_without_ready_reliability(self):
        for rel in (None, {}, {"matched": 0, "logged": 33},
                    {"n": 1, "direction_hit_rate": 1, "median_abs_error_pts": 2}):
            with self.subTest(reliability=rel):
                result = ia.analyze_ipo({"gmp": 10, "price": 100, "est_gain_pct": 10}, reliability=rel)
                self.assertTrue(any("not enough matched" in risk for risk in result["risks"]))

    def test_analysis_with_ready_reliability(self):
        result = ia.analyze_ipo({"gmp": 10, "price": 100, "est_gain_pct": 10},
                                reliability={"n": 10, "direction_hit_rate": .8, "median_abs_error_pts": 5})
        self.assertTrue(any("80% of 10" in risk for risk in result["risks"]))

    def test_board_names_status_counts_and_links(self):
        html = '''<h2>Mainboard</h2><table>
        <tr><th>Company</th><th>GMP</th></tr>
        <tr><td><a href="/jio-ipo/">Jio Platforms Open</a></td><td>10</td></tr>
        <tr><td>Example Upcoming</td><td>0</td></tr>
        <tr><td>Legacy (O)</td><td>5</td></tr>
        <tr><td>Finished Closed</td><td>5</td></tr>
        <tr><td>Trading Listed</td><td>5</td></tr>
        </table>'''
        with patch.object(ia.requests, "get", return_value=Mock(text=html)):
            board = ia.fetch_gmp_tables()["mainboard"]
        self.assertEqual(board["name"].tolist(), ["Jio Platforms", "Example", "Legacy", "Finished", "Trading"])
        self.assertEqual(board["status"].tolist(), ["Open", "Upcoming", "Open", "Closed", "Listed"])
        self.assertEqual((board["status"] == "Open").sum(), 2)
        self.assertEqual(board.iloc[0]["link"], "https://ipowatch.in/jio-ipo/")

    def test_explicit_status_is_preserved(self):
        board = ia._parse_live_board(pd.DataFrame({
            "Company": ["Example Open", "Other Upcoming", "Firm (India)"],
            "GMP": [1, 2, 3], "Status": ["Closed", " ", None]}), "mainboard")
        self.assertEqual(board["status"].iloc[:2].tolist(), ["Closed", "Upcoming"])
        self.assertEqual(board.iloc[2]["name"], "Firm (India)")

    def test_detail_resolves_relative_and_absolute_links(self):
        for url in ("/jio-ipo/", "https://ipowatch.in/jio-ipo/"):
            with self.subTest(url=url), patch.object(ia.requests, "get", return_value=Mock(text="<title>IPO</title>")) as get:
                detail = ia.fetch_ipo_detail(url)
                self.assertEqual(get.call_args.args[0], "https://ipowatch.in/jio-ipo/")
                self.assertEqual(detail["url"], "https://ipowatch.in/jio-ipo/")


if __name__ == "__main__":
    unittest.main()
