"""
backtest.py
-----------
Turns a model's price predictions into Buy/Sell/Hold decisions and
simulates the resulting portfolio value, so you can compare each
model's real trading performance against a Buy-and-Hold baseline.

Strategy (simple, easy to explain in the report):
    - Compare predicted next-day price to today's actual close.
    - predicted % change > +buy_threshold   -> BUY  (go long / stay long)
    - predicted % change < -sell_threshold  -> SELL (exit to cash)
    - otherwise                             -> HOLD (keep current position)
"""

import numpy as np
import pandas as pd


def simulate_strategy(
    actual_close: pd.Series,
    predicted_close: pd.Series,
    initial_investment: float = 10_000.0,
    buy_threshold: float = 0.01,   # +1%
    sell_threshold: float = 0.01,  # -1%
    commission_pct: float = 0.0,   # e.g. 0.001 = 0.1% per trade -- realistic brokerage commission
    slippage_pct: float = 0.0,     # e.g. 0.0005 = 0.05% per trade -- price you actually get vs. quoted price
):
    """
    Simulate a long/flat trading strategy over the test period.

    Parameters
    ----------
    actual_close : pd.Series
        Actual closing prices for the test period (chronological, indexed by date).
    predicted_close : pd.Series
        Model's predicted next-day closing prices, same length/index as actual_close.
    initial_investment : float
        Starting cash.
    buy_threshold, sell_threshold : float
        % move thresholds that trigger a buy/sell signal.
    commission_pct, slippage_pct : float
        Real-world trading frictions, as a fraction (not %) of trade value,
        deducted on EVERY buy and sell. Default 0 preserves the original
        frictionless backtest behavior used elsewhere in this project.
        For real-use analysis, try commission_pct=0.001 (0.1%, a typical
        discount broker) and slippage_pct=0.0005 (0.05%) to see how much
        of your paper profit a strategy with many trades would actually
        lose to real-world costs -- this matters a lot for high-trade-count
        strategies (see the SVR overtrading example in the README).

    Returns
    -------
    dict with summary results, and a DataFrame with day-by-day portfolio value.
    """
    actual_close = actual_close.reset_index(drop=True)
    predicted_close = predicted_close.reset_index(drop=True)
    total_cost_pct = commission_pct + slippage_pct

    cash = initial_investment
    shares = 0
    in_position = False
    num_trades = 0
    total_costs_paid = 0.0

    portfolio_values = []
    actions = []

    for i in range(len(actual_close) - 1):
        today_price = actual_close.iloc[i]
        predicted_next = predicted_close.iloc[i]
        predicted_pct_change = (predicted_next - today_price) / today_price

        action = "HOLD"

        if predicted_pct_change > buy_threshold and not in_position:
            # BUY: go all-in, minus trading costs
            trade_value = cash
            cost = trade_value * total_cost_pct
            shares = (cash - cost) / today_price
            cash = 0
            in_position = True
            num_trades += 1
            total_costs_paid += cost
            action = "BUY"

        elif predicted_pct_change < -sell_threshold and in_position:
            # SELL: exit to cash, minus trading costs
            trade_value = shares * today_price
            cost = trade_value * total_cost_pct
            cash = trade_value - cost
            shares = 0
            in_position = False
            num_trades += 1
            total_costs_paid += cost
            action = "SELL"

        # Mark-to-market portfolio value at today's price
        portfolio_value = cash + shares * today_price
        portfolio_values.append(portfolio_value)
        actions.append(action)

    # Liquidate any open position at the final price for a clean comparison
    final_price = actual_close.iloc[-1]
    if in_position:
        trade_value = shares * final_price
        cost = trade_value * total_cost_pct
        final_value = trade_value - cost
        total_costs_paid += cost
    else:
        final_value = cash
    portfolio_values.append(final_value)
    actions.append("FINAL_LIQUIDATE" if in_position else "HOLD")

    total_profit = final_value - initial_investment
    pct_return = (total_profit / initial_investment) * 100

    daily_log = pd.DataFrame({
        "Actual_Close": actual_close.values,
        "Predicted_Next_Close": predicted_close.values,
        "Action": actions,
        "Portfolio_Value": portfolio_values,
    })

    summary = {
        "Initial_Investment": initial_investment,
        "Final_Value": round(final_value, 2),
        "Total_Profit_Loss": round(total_profit, 2),
        "Percent_Return": round(pct_return, 2),
        "Num_Trades": num_trades,
        "Total_Costs_Paid": round(total_costs_paid, 2),
    }

    return summary, daily_log


def buy_and_hold_baseline(actual_close: pd.Series, initial_investment: float = 10_000.0):
    """The baseline every model-based strategy must be compared against."""
    actual_close = actual_close.reset_index(drop=True)
    start_price = actual_close.iloc[0]
    end_price = actual_close.iloc[-1]

    shares = initial_investment / start_price
    final_value = shares * end_price
    total_profit = final_value - initial_investment
    pct_return = (total_profit / initial_investment) * 100

    return {
        "Initial_Investment": initial_investment,
        "Final_Value": round(final_value, 2),
        "Total_Profit_Loss": round(total_profit, 2),
        "Percent_Return": round(pct_return, 2),
        "Num_Trades": 1,
        "Total_Costs_Paid": 0.0,
    }
