import json
import unittest
from day_trader_runtime import Workspace, ROOT, TOOLS, symbols


class DayTraderTests(unittest.TestCase):
    def test_all_bundled_tools_have_working_cli(self):
        ws = Workspace()
        for tool in TOOLS:
            if tool == 'Broker setup':
                continue
            with self.subTest(tool=tool):
                code, output = ws.run(tool, ['--help'])
                self.assertEqual(code, 0, output)

    def test_paper_plan_sizing_and_risk_pipeline_is_isolated(self):
        ws, other = Workspace(), Workspace()
        code, output = ws.run('Exit planner', ['plan', '--symbol', 'TSLA', '--entry', '100',
            '--setup', 'level', '--stop', 'price:98', '--target', '2R', '--date', '2026-10-19',
            '--time', '10:00', '--interval', '5m', '--now', '2026-10-19 10:00', '--write'])
        self.assertEqual(code, 0, output)
        self.assertEqual(ws.read('plan.json')['stop'], 98)
        self.assertIsNone(other.read('plan.json'))
        config = json.loads((ROOT / 'trader-budget/config.example.json').read_text())
        config.pop('sec_user_agent')
        ws.write('config.json', config)
        code, output = ws.run('Budget', ['--limit', '--write', '--now', '2026-10-19 10:00'])
        self.assertEqual(code, 0, output)
        code, output = ws.run('Risk check', ['--now', '2026-10-19 10:00'])
        self.assertEqual(code, 0, output)
        self.assertIn('PASS', output)
        self.assertIsNone(other.read('orders.json'))
        ws.invalidate('orders.json', 'state/risk.json')
        self.assertIsNone(ws.read('orders.json'))

    def test_symbol_validation_and_broker_action_boundary(self):
        for value in ['IDEA.NS', '../file', '--help', 'AAPL;echo x', '']:
            with self.assertRaises(ValueError):
                symbols(value)
        self.assertEqual(symbols('aapl, MSFT, aapl'), ['AAPL', 'MSFT'])
        with self.assertRaises(ValueError):
            Workspace().run('Broker setup', ['approve'])

    def test_all_screens_render_without_network_or_training(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('import day_trader_page\nday_trader_page.render()', default_timeout=20)
        for tool in TOOLS:
            app.session_state['day_trader_tool'] = tool
            app.run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.title[0].value, 'Day Trader')


if __name__ == '__main__':
    unittest.main()
