"""Build the paper-trade workbook for the profitable week (16-22 Sep 2026).

Every result cell is a formula referencing the Assumptions sheet, so changing the
margin rate, lot size or cost per lot recalculates the whole workbook.

Only Excel-2007-era functions are used (SUM, SUMIF, COUNTIF, MAX, MIN, IFERROR)
because no LibreOffice is available in this environment to verify newer ones.
The week's trades are a contiguous block, so plain ranges do the job and no
array or *IFS formula is needed.
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

SRC = "paper_trade_export.csv"
OUT = "NSE_short_straddle_paper_trade.xlsx"
WEEK = "Week 1 (16-22 Sep)"

FONT = "Arial"
BLUE = Font(name=FONT, size=10, color="0000FF")
BLACK = Font(name=FONT, size=10)
GREEN = Font(name=FONT, size=10, color="008000")
HDR = Font(name=FONT, size=10, bold=True, color="FFFFFF")
TITLE = Font(name=FONT, size=14, bold=True)
SUB = Font(name=FONT, size=10, italic=True, color="595959")
BOLD = Font(name=FONT, size=10, bold=True)
BIG = Font(name=FONT, size=18, bold=True, color="1F7A1F")
HDR_FILL = PatternFill("solid", fgColor="1F3864")
YELLOW = PatternFill("solid", fgColor="FFFF00")
BAND = PatternFill("solid", fgColor="F2F2F2")
GREENF = PatternFill("solid", fgColor="E2EFDA")
REDF = PatternFill("solid", fgColor="FCE4E4")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

RS = '₹#,##0;(₹#,##0);-'
RS2 = '₹#,##0.00;(₹#,##0.00);-'
PCT = '0.00%;(0.00%);-'
NUM = '#,##0;(#,##0);-'


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HDR
        cell.fill = HDR_FILL
        cell.border = BOX
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 30


def widths(ws, spec):
    for col, w in spec.items():
        ws.column_dimensions[col].width = w


def bullets(ws, start_row, ncols, items, height=28):
    for i, n in enumerate(items, start=start_row):
        c = ws.cell(row=i, column=1, value="• " + n)
        c.font = BLACK
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=ncols)
        ws.row_dimensions[i].height = height


def main():
    df = pd.read_csv(SRC, parse_dates=["date", "expiry"])
    df = df[df.week == WEEK].sort_values(["date", "symbol", "cp"]).reset_index(drop=True)
    wb = Workbook()

    # ---------------------------------------------------------- Assumptions
    a = wb.active
    a.title = "Assumptions"
    a["A1"] = "Short ATM Straddle — Assumptions & Method"
    a["A1"].font = TITLE
    a["A2"] = ("Yellow cells drive every figure in this workbook. "
               "Change one and the model recalculates.")
    a["A2"].font = SUB
    rows = [
        ("Parameter", "Value", "Unit", "Source / note"),
        ("Margin requirement", 0.12, "of notional",
         "SPAN + exposure on a short index option, approximate. Broker-specific — confirm with yours."),
        ("Cost per lot (round trip)", 60, "₹",
         "Brokerage + STT + exchange fees + GST + stamp duty, one lot in and out, discount broker."),
        ("NIFTY lot size", 75, "units", "NSE contract specification."),
        ("BANKNIFTY lot size", 35, "units", "NSE contract specification (no BANKNIFTY trades this week)."),
        ("Max days to expiry", 7, "days", "Strategy rule — weekly contracts only."),
        ("ATM band", 0.005, "of spot", "Strike within ±0.5% of the PRIOR session close."),
        ("Min prior-session volume", 1000, "contracts",
         "Liquidity filter on the PREVIOUS session. Never same-day volume, which is look-ahead."),
    ]
    for i, r in enumerate(rows, start=4):
        for j, v in enumerate(r, start=1):
            c = a.cell(row=i, column=j, value=v)
            c.border = BOX
            c.font = HDR if i == 4 else BLACK
            if i > 4 and j == 2:
                c.font = BLUE
                c.fill = YELLOW
            if j == 4:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    style_header(a, 4, 4)
    a["B5"].number_format = "0.0%"
    a["B10"].number_format = "0.000%"
    widths(a, {"A": 28, "B": 14, "C": 14, "D": 74})

    a["A14"] = "Method"
    a["A14"].font = BOLD
    bullets(a, 15, 4, [
        "At the open, SELL the at-the-money call and the at-the-money put.",
        "Strike chosen from the PRIOR session close — the only spot a trader knows at the open.",
        "Liquidity filtered on the PRIOR session volume, never the current day's.",
        "Cover both legs at the close. No position held overnight.",
        "Prices are real NSE exchange data (daily F&O bhavcopy), not modelled.",
    ], height=16)

    a["A21"] = "Data integrity — three look-ahead leaks were removed before these numbers"
    a["A21"].font = BOLD
    bullets(a, 22, 4, [
        "Selecting contracts by SAME-DAY volume, which is only known at the close. Worth 25x in P&L.",
        "UndrlygPric in the bhavcopy is the CLOSING spot (verified at 0.0000% difference against the "
        "NIFTY close), so choosing 'ATM' with it picks the strike that ENDED at the money — exactly "
        "the strike that decayed most that day.",
        "Testing only a calm window. With the leaks present the backtest showed Sharpe 17.51; "
        "corrected, it is Sharpe 1.27.",
    ])

    # ---------------------------------------------------------- Trade Log
    t = wb.create_sheet("Trade Log")
    t["A1"] = "Trade Log — " + WEEK
    t["A1"].font = TITLE
    t["A2"] = "Real NSE exchange prices. Blue = raw exchange data, black = formula, green = link to Assumptions."
    t["A2"].font = SUB
    hdr = ["Date", "Symbol", "C/P", "Expiry", "DTE", "Strike", "Prior close", "Spot (close)",
           "Sell @ open", "Buy @ close", "Points", "Lot", "Gross ₹", "Cost ₹",
           "Net P&L ₹", "Margin ₹", "% of margin"]
    for j, h in enumerate(hdr, start=1):
        t.cell(row=4, column=j, value=h)
    style_header(t, 4, len(hdr))

    first = 5
    last = first + len(df) - 1
    for i, r in enumerate(df.itertuples(), start=first):
        t.cell(row=i, column=1, value=r.date).number_format = "dd-mmm-yyyy"
        t.cell(row=i, column=2, value=r.symbol)
        t.cell(row=i, column=3, value=r.cp)
        t.cell(row=i, column=4, value=r.expiry).number_format = "dd-mmm-yyyy"
        t.cell(row=i, column=5, value=int(r.dte))
        t.cell(row=i, column=6, value=float(r.strike)).number_format = NUM
        t.cell(row=i, column=7, value=float(r.prev_spot)).number_format = "#,##0.00"
        t.cell(row=i, column=8, value=float(r.spot)).number_format = "#,##0.00"
        t.cell(row=i, column=9, value=float(r.open)).number_format = RS2
        t.cell(row=i, column=10, value=float(r.close)).number_format = RS2
        t.cell(row=i, column=11, value=f"=I{i}-J{i}").number_format = "#,##0.00"
        t.cell(row=i, column=12, value=int(r.lot))
        t.cell(row=i, column=13, value=f"=K{i}*L{i}").number_format = RS
        t.cell(row=i, column=14, value="=Assumptions!$B$6").number_format = RS
        t.cell(row=i, column=15, value=f"=M{i}-N{i}").number_format = RS
        t.cell(row=i, column=16, value=f"=G{i}*L{i}*Assumptions!$B$5").number_format = RS
        t.cell(row=i, column=17, value=f"=IFERROR(O{i}/P{i},0)").number_format = PCT
        for j in range(1, len(hdr) + 1):
            t.cell(row=i, column=j).border = BOX
            t.cell(row=i, column=j).font = BLACK
        for j in (6, 7, 8, 9, 10, 12):
            t.cell(row=i, column=j).font = BLUE
        for j in (14, 16):
            t.cell(row=i, column=j).font = GREEN
        t.cell(row=i, column=15).fill = GREENF if (r.open - r.close) * r.lot - 60 > 0 else REDF

    tot = last + 1
    t.cell(row=tot, column=1, value="TOTAL").font = BOLD
    for col in (13, 14, 15):
        L = get_column_letter(col)
        c = t.cell(row=tot, column=col, value=f"=SUM({L}{first}:{L}{last})")
        c.number_format = RS
        c.font = BOLD
        c.border = BOX
    t.cell(row=tot, column=1).border = BOX
    t.freeze_panes = "A5"
    widths(t, {"A": 13, "B": 11, "C": 6, "D": 13, "E": 6, "F": 10, "G": 12, "H": 13,
               "I": 12, "J": 12, "K": 10, "L": 7, "M": 12, "N": 10, "O": 13,
               "P": 13, "Q": 12})

    # ---------------------------------------------------------- Daily
    d = wb.create_sheet("Daily")
    d["A1"] = "Daily Summary — " + WEEK
    d["A1"].font = TITLE
    dh = ["Date", "Positions", "Net P&L ₹", "Margin ₹", "% of margin", "Cumulative ₹"]
    for j, h in enumerate(dh, start=1):
        d.cell(row=3, column=j, value=h)
    style_header(d, 3, len(dh))

    days = sorted(df.date.unique())
    r0 = 4
    for i, dt in enumerate(days, start=r0):
        c = d.cell(row=i, column=1, value=pd.Timestamp(dt))
        c.number_format = "dd-mmm-yyyy"
        c.font = BLACK
        rng = f"'Trade Log'!$A${first}:$A${last}"
        d.cell(row=i, column=2, value=f"=COUNTIF({rng},$A{i})").number_format = NUM
        d.cell(row=i, column=3,
               value=f"=SUMIF({rng},$A{i},'Trade Log'!$O${first}:$O${last})").number_format = RS
        d.cell(row=i, column=4,
               value=f"=SUMIF({rng},$A{i},'Trade Log'!$P${first}:$P${last})").number_format = RS
        d.cell(row=i, column=5, value=f"=IFERROR(C{i}/D{i},0)").number_format = PCT
        d.cell(row=i, column=6, value=f"=SUM($C${r0}:C{i})").number_format = RS
        for j in range(1, len(dh) + 1):
            d.cell(row=i, column=j).border = BOX
            if j >= 2:
                d.cell(row=i, column=j).font = BLACK
    dlast = r0 + len(days) - 1
    dt_row = dlast + 1
    d.cell(row=dt_row, column=1, value="TOTAL").font = BOLD
    for col in (2, 3):
        L = get_column_letter(col)
        c = d.cell(row=dt_row, column=col, value=f"=SUM({L}{r0}:{L}{dlast})")
        c.number_format = NUM if col == 2 else RS
        c.font = BOLD
        c.border = BOX
    d.cell(row=dt_row, column=1).border = BOX
    widths(d, {"A": 14, "B": 11, "C": 14, "D": 14, "E": 13, "F": 15})
    d.freeze_panes = "A4"

    # ---------------------------------------------------------- Summary
    s = wb.create_sheet("Summary", 0)
    s["A1"] = "NSE Short ATM Straddle — Paper Trade"
    s["A1"].font = TITLE
    s["A2"] = (WEEK + " · NIFTY · intraday (sell at open, cover at close) "
               "· real NSE F&O exchange prices")
    s["A2"].font = SUB

    P = f"'Trade Log'!$O${first}:$O${last}"
    s["A4"] = "NET PROFIT"
    s["A4"].font = BOLD
    s["B4"] = f"=SUM({P})"
    s["B4"].font = BIG
    s["B4"].number_format = RS
    s["C4"] = "on peak margin of"
    s["C4"].font = SUB
    s["D4"] = f"=MAX(Daily!$D${r0}:$D${dlast})"
    s["D4"].number_format = RS
    s["D4"].font = BOLD
    s["E4"] = f"=IFERROR(B4/D4,0)"
    s["E4"].number_format = PCT
    s["E4"].font = BIG
    s.row_dimensions[4].height = 26

    for j, h in enumerate(["Metric", "Value"], start=1):
        s.cell(row=6, column=j, value=h)
    style_header(s, 6, 2)
    metrics = [
        ("Trades", f"=COUNT({P})", NUM),
        ("Winning trades", f'=COUNTIF({P},">0")', NUM),
        ("Losing trades", f'=COUNTIF({P},"<=0")', NUM),
        ("Win rate", "=IFERROR(B8/B7,0)", PCT),
        ("Sessions traded", f"=COUNT(Daily!$C${r0}:$C${dlast})", NUM),
        ("Winning sessions", f'=COUNTIF(Daily!$C${r0}:$C${dlast},">0")', NUM),
        ("Gross P&L ₹", f"=SUM('Trade Log'!$M${first}:$M${last})", RS),
        ("Total costs ₹", f"=SUM('Trade Log'!$N${first}:$N${last})", RS),
        ("NET P&L ₹", f"=SUM({P})", RS),
        ("Best trade ₹", f"=MAX({P})", RS),
        ("Worst trade ₹", f"=MIN({P})", RS),
        ("Average trade ₹", f"=IFERROR(AVERAGE({P}),0)", RS),
        ("Peak margin ₹", f"=MAX(Daily!$D${r0}:$D${dlast})", RS),
        ("Return on peak margin", "=IFERROR(B15/B19,0)", PCT),
    ]
    for i, (name, f, fmt) in enumerate(metrics, start=7):
        s.cell(row=i, column=1, value=name).font = BOLD
        c = s.cell(row=i, column=2, value=f)
        c.number_format = fmt
        c.font = BLACK
        for j in (1, 2):
            s.cell(row=i, column=j).border = BOX
            if i % 2 == 0:
                s.cell(row=i, column=j).fill = BAND

    s["A22"] = "What happened this week"
    s["A22"].font = BOLD
    bullets(s, 23, 5, [
        "NIFTY barely moved: 23,201.60 to 23,329.00 across the whole week, +0.55%, "
        "with a largest single-day move of 0.36%.",
        "A short straddle is a bet on stillness, not a forecast. It collects premium when "
        "price sits still, so a near-motionless week is close to its best case.",
        "All five sessions were positive and 7 of the 10 legs made money.",
        "CONTEXT — this is one selected week from an eight-month record. The following week "
        "(23-29 Sep) lost ₹39,140 on the same rules when the market fell 2-4%. Over the "
        "eight-month backtest the strategy earned ₹45,470 with a maximum drawdown of ₹94,210.",
    ])

    s["A28"] = ("NOT INVESTMENT ADVICE. A paper trade on historical exchange data. "
                "Real fills, slippage and margin calls will differ from these figures.")
    s["A28"].font = Font(name=FONT, size=10, bold=True, color="C00000")
    s.merge_cells("A28:E28")
    widths(s, {"A": 26, "B": 18, "C": 18, "D": 16, "E": 12})

    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False

    wb.save(OUT)
    print("wrote", OUT)
    print(f"  trade rows {first}-{last} ({len(df)} trades), daily rows {r0}-{dlast}")


if __name__ == "__main__":
    main()
