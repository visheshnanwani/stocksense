"""On-demand Day Trader workbench; no model training on entry."""
import datetime as dt
import json
from zoneinfo import ZoneInfo
import streamlit as st
from day_trader_runtime import ROOT, TOOLS, Workspace, symbols


def render():
    st.title('Day Trader')
    st.caption('Scan → check news → plan an entry and exit → simulate → size → review risk.')
    st.info('US-listed stocks · USD ($) · New York market time. NSE/BSE analysis remains in Stock analysis. '
            'Data can be delayed. These tools prepare research and paper-trade plans; they do not place orders.')
    if 'day_trader_workspace' not in st.session_state:
        st.session_state.day_trader_workspace = Workspace()
    ws = st.session_state.day_trader_workspace
    tool = st.selectbox('Tool', list(TOOLS), key='day_trader_tool')
    st.caption('Plans and results belong to this browser session. Download anything you want to keep; '
               'a new session or server restart clears them. Tools run only when you click a button.')
    identity = ''
    if tool in ('News', 'Earnings'):
        identity = st.text_input('SEC identity (your name and email)', key='day_trader_sec',
                                 help='Sent to SEC EDGAR as its required User-Agent only when you run a filings request.')
    args = None
    invalidate = []
    try:
        if tool == 'Scanner':
            tickers = st.text_input('US symbols (up to 20)', 'AAPL, MSFT, NVDA, TSLA')
            mode = st.radio('Scan', ['Daily setups', 'Opening-range breakouts'], horizontal=True)
            floor = st.number_input('Minimum price ($)', min_value=0.0, value=5.0)
            st.caption('Matches rules, not buy recommendations. Scans only the symbols you enter.')
            if st.button('Scan symbols', type='primary'):
                args = ['daily' if mode == 'Daily setups' else 'orb', '--symbols', ','.join(symbols(tickers)),
                        '--min-price', floor, '--show', '20']
                invalidate = ['watchlist.json']
        elif tool == 'News':
            tickers = st.text_input('US symbols', 'AAPL')
            hours = st.slider('Look back (hours)', 1, 72, 24)
            filings = st.checkbox('Include SEC filings', value=False)
            st.caption('Headlines link to source reporting; they are not independently verified facts. Refresh manually for new items.')
            if st.button('Check news', type='primary'):
                args = ['--tickers', *symbols(tickers, 5), '--hours', hours]
                if not filings:
                    args += ['--no-sec']
                    identity = ''
                elif '@' not in identity:
                    raise ValueError('Enter your real name and email for SEC filings, or turn off Include SEC filings.')
        elif tool == 'Entry timer':
            ticker = st.text_input('US symbol', 'AAPL')
            setup = st.radio('Entry rule', ['Opening range', 'Price level'], horizontal=True)
            level = st.number_input('Breakout level ($)', min_value=0.01, value=100.0) if setup == 'Price level' else None
            st.caption('Checks completed five-minute candles. Click again to refresh; no background watcher is started.')
            if st.button('Check entry', type='primary'):
                args = ['check', symbols(ticker, 1)[0], '--write']
                if level is not None:
                    args += ['--setup', 'level', '--level', level]
                invalidate = ['entry.json']
        elif tool == 'Exit planner':
            source = st.radio('Plan from', ['Saved entry signal', 'Manual prices'], horizontal=True)
            ticker = st.text_input('US symbol', 'AAPL') if source == 'Manual prices' else None
            entry = st.number_input('Entry ($)', min_value=0.01, value=100.0) if ticker else None
            stop = st.number_input('Stop ($)', min_value=0.01, value=98.0)
            target_r = st.number_input('Target as a multiple of risk (R)', min_value=0.1, value=2.0)
            today = dt.datetime.now(ZoneInfo('America/New_York')).date()
            date = st.date_input('Plan date (New York)', today)
            clock = st.time_input('Entry candle close (New York, five-minute boundary)', dt.time(10, 0), step=300)
            st.caption('Long positions only. Stops can fill worse than the planned price. Manual prices are hypothetical.')
            if st.button('Save exit plan', type='primary'):
                args = ['plan', '--stop', f'price:{stop}', '--target', f'{target_r}R', '--write']
                if ticker:
                    args += ['--symbol', symbols(ticker, 1)[0], '--entry', entry, '--setup', 'level',
                             '--date', str(date), '--time', clock.strftime('%H:%M'), '--interval', '5m']
                elif not ws.read('entry.json'):
                    raise ValueError('Run Entry timer first, or choose Manual prices.')
                invalidate = ['plan.json', 'orders.json', 'timemachine.json', 'state/risk.json', 'alerts.json']
        elif tool == 'Trade simulation':
            st.caption('Replays historical five-minute bars. Simulated odds are not forecasts. '
                       'Requires at least 20 complete prior sessions and a saved plan with a completed entry candle.')
            replays = st.select_slider('Simulated sessions', [1000, 5000, 10000], value=1000)
            seed = st.number_input('Random seed', min_value=0, value=19)
            if st.button('Simulate saved plan', type='primary'):
                if not ws.read('plan.json'):
                    raise ValueError('Save an exit plan first.')
                args = ['run', '--replays', replays, '--seed', seed, '--until', '15:55', '--write']
                invalidate = ['timemachine.json']
        elif tool == 'Budget':
            st.caption('Set a hypothetical paper account and its limits. Saving these values does not fund a brokerage account.')
            budget = st.number_input('Paper account budget ($)', min_value=1.0, value=5000.0)
            loss = st.number_input('Maximum planned loss per trade ($)', min_value=0.01, value=50.0)
            daily = st.number_input('Daily risk limit ($)', min_value=0.01, value=150.0)
            position = st.slider('Maximum position (% of budget)', 1, 100, 25)
            buffer = st.slider('Cash buffer (%)', 0, 99, 10)
            order = st.number_input('Maximum order ($)', min_value=1.0, value=1000.0)
            trades = st.number_input('Maximum trades per day', min_value=1, max_value=50, value=3)
            rr = st.number_input('Minimum reward:risk', min_value=0.1, value=1.0)
            if st.button('Save limits and size plan', type='primary'):
                if not ws.read('plan.json'):
                    raise ValueError('Save an exit plan first.')
                config = json.loads((ROOT / 'trader-budget/config.example.json').read_text())
                config.pop('sec_user_agent', None)
                config.update(budget=budget, max_loss_per_trade_usd=loss, max_daily_loss_usd=daily,
                              max_position_pct=position, cash_buffer_pct=buffer, max_order_usd=order,
                              max_trades_per_day=trades, min_reward_risk=rr, paper=True)
                ws.write('config.json', config)
                args = ['--limit', '--write']
                invalidate = ['orders.json', 'state/risk.json']
        elif tool == 'Risk check':
            st.caption('Checks the saved proposal against your paper-account limits, latest saved quote and US market window. '
                       'PASS means rules passed; it is not a recommendation or order approval.')
            if st.button('Check saved order', type='primary'):
                if not ws.read('orders.json') or not ws.read('config.json'):
                    raise ValueError('Use Budget to save your limits and size a plan first.')
                args = []
                invalidate = ['state/risk.json']
        elif tool == 'Broker setup':
            st.subheader('Paper order preview')
            st.write('Create a plan, size it in Budget, then run Risk check. This page shows the resulting ticket; '
                     'it is not connected to a broker and cannot submit or approve orders.')
            st.link_button('Alpaca paper account setup', 'https://docs.alpaca.markets/docs/paper-trading')
            st.link_button('Official Alpaca MCP connector', 'https://github.com/alpacahq/alpaca-mcp-server')
            st.caption('Keep paper credentials in your local broker connector, not this public dashboard. '
                       'Review and submit any order yourself in your broker’s interface.')
            if st.button('Preview saved ticket', type='primary'):
                if not ws.read('orders.json'):
                    raise ValueError('Create an order proposal in Budget first.')
                args = ['show']
        elif tool == 'Alerts':
            st.caption('Manual checks only: alerts do not run after you leave this page or while the free server sleeps. '
                       'No email, push notification or automatic trading is enabled.')
            action = st.selectbox('Action', ['Check alerts', 'Add stop/target alerts from plan', 'List alerts', 'Get quote', 'Market clock'])
            ticker = st.text_input('US symbol', 'AAPL') if action == 'Get quote' else None
            if st.button('Run alert action', type='primary'):
                args = {'Check alerts': ['check'], 'Add stop/target alerts from plan': ['add', '--from-plan'],
                        'List alerts': ['list'], 'Market clock': ['clock']}.get(action)
                if ticker:
                    args = ['quote', symbols(ticker, 1)[0], '--write']
                if action == 'Add stop/target alerts from plan' and not ws.read('plan.json'):
                    raise ValueError('Save an exit plan first.')
        elif tool == 'Earnings':
            ticker = st.text_input('US symbol', 'AAPL')
            st.caption('Reads reported SEC figures and filing links. SEC data can lag; derived quarters are identified by the tool.')
            if st.button('Read earnings filings', type='primary'):
                if '@' not in identity or len(identity.split()) < 2:
                    raise ValueError('Enter your real name and email for the SEC User-Agent.')
                args = [symbols(ticker, 1)[0], '--quarters', '8', '--metrics', 'trader/metrics.json']
                invalidate = ['metrics.json']
        elif tool == 'Congress disclosures':
            member = st.text_input('House member name', 'Pelosi')
            year = st.number_input('Filing year', min_value=2008, max_value=dt.date.today().year, value=dt.date.today().year)
            st.caption('Public House disclosures can appear weeks after a trade. They do not show current holdings or future intent. '
                       'Open the original filing to inspect its transaction rows; Senate filings are separate.')
            if st.button('Find disclosures', type='primary'):
                if not member.strip() or len(member) > 100 or member.startswith('-'):
                    raise ValueError('Enter a member name.')
                args = ['index', '--member', member.strip(), '--year', year, '--limit', '20']
        if args is not None:
            ws.invalidate(*invalidate)
            with st.spinner(f'Running {tool.lower()}…'):
                code, output = ws.run(tool, args, sec_identity=identity)
            st.session_state[f'dt_result_{tool}'] = (code, output, dt.datetime.now(dt.timezone.utc).isoformat())
    except (ValueError, OSError) as exc:
        st.error(str(exc))
    result = st.session_state.get(f'dt_result_{tool}')
    if result:
        code, output, timestamp = result
        st.caption(f'Last run: {timestamp} · exit status {code}')
        if code:
            st.warning('The tool returned a refusal, no matches, or a data error. Read the details below.')
        st.code(output, language=None, wrap_lines=True)
    with st.expander('Saved plans and results'):
        for name in ('watchlist.json', 'entry.json', 'plan.json', 'orders.json', 'timemachine.json', 'metrics.json', 'alerts.json', 'state/risk.json'):
            value = ws.read(name)
            if value is not None:
                st.download_button(f'Download {name}', json.dumps(value, indent=2),
                                   file_name=name.replace('/', '-'), mime='application/json', key=f'dt_download_{name}')
    st.caption('Powered by Jakeschincariol/day-trader-skill · MIT license · Research tools, not investment advice.')
