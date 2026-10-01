# Stock Market Prediction & Profitability — Model Comparison Project

Compares 8 models on next-day closing-price forecasting for a chosen
stock, and backtests a trading strategy built on each model's
predictions against a Buy-and-Hold baseline.

**Models covered:** Linear Regression, Random Forest, SVR, XGBoost,
ARIMA, LSTM, GRU, Transformer.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

1. Open `main.py` and edit the CONFIG block at the top:
   - `TICKER` — leave as `None` and the script will **ask you for a stock
     name when it runs** (type e.g. `apple`, `reliance`, `tata motors`,
     pick from the list of real matching stocks). Or set a ticker directly
     (e.g. `"AAPL"`, `"RELIANCE.NS"`) to skip the prompt. Every CLI script
     (`ensemble.py`, `predict_future.py`, `train_advanced.py`, ...) works
     the same way.
   - `START_DATE` / `END_DATE` — time period to analyze
   - `EPOCHS` — training epochs for LSTM/GRU/Transformer (start at 20,
     increase to 50–100 once you're confident it runs cleanly)

2. Run it:
   ```bash
   python main.py
   ```

3. Outputs:
   - `forecasting_metrics_comparison.csv` — MAE/MSE/RMSE/MAPE/R² for all 8 models
   - `profitability_comparison.csv` — profit/loss, % return, trades vs Buy-and-Hold

Both tables print to the console too.

## Three reported bugs (v45)

### 1. The loading bar froze at 6% in the IPO Center

The IPO branch runs `ipo_page.render()` and then `st.stop()`, so nothing after
it ever executed -- and every `_load()` call lives after it. The bar was set to
6% at startup and that was the last thing to touch it.

`ipo_page.render()` now takes an optional `on_progress` callback and reports its
own stages (12% loading the board, 55% scoring GMP reliability, 72% building the
board, 100% done), with the host passing `_load` in. Verified: the bar reaches
100%, takes the `done` class and fades to opacity 0 while the page renders.

### 2. The sidebar could not be scrolled

Measured rather than guessed. The sidebar had **no scroller at all**:

    [data-testid="stSidebar"]      height 2294px   (viewport 768px)
    its inner div                  clientH 2294, scrollH 2294  -> nothing to scroll
    [data-testid="stAppViewContainer"]  height 2294px
    .stApp                         height 768px, overflow-y hidden

With no height constraint the sidebar grew to its content, the flex container
grew with it, and `.stApp` clipped the overflow -- so everything past the fold
was simply unreachable, and a wheel event had nowhere to go. The only scroller
on the page was `SECTION[stMain]`.

Pinning the view container and the sidebar to `100vh` and letting the sidebar's
inner div own the overflow gives it something to scroll. Verified after the fix:
sidebar 768px, inner scrollHeight 2294, `scrollTop` moves.

My first guess was wrong and worth recording: I assumed the FX rule
`.stApp > * { position: relative; z-index: 1 }` was the cause. It is not -- the
sidebar is a child of `stAppViewContainer`, not of `.stApp`, and removing the
rule left the height at 2294. The height was never about positioning.

### 3. Refresh rate

Options were `[2, 5, 10, 30, 60]` seconds, defaulting to 5. Now
`[1, 2, 3, 5, 10, 30, 60]`, defaulting to **2 seconds**, with the three fallback
reads of `live_refresh_s` updated to match.

A caveat worth knowing: each tick re-fetches the live quote from Yahoo, which is
itself only updated every 5-15 seconds. Setting 1s polls faster than the data
changes and mostly spends requests -- 2s is the useful floor.

## The 2.5x stop, shipped and re-run (v44)

The stop is now the default in `exp_real_options.py` (`STOP_MULT = 2.5`,
`STOP_SLIP_PCT = 5.0` -- a leg is covered when its intraday high reaches 2.5x
what it was sold at, filled 5% worse than the trigger).

### Over the full history it is a clear gain

    8 months, no stop     +Rs 45,470   Sharpe 1.27
    8 months, 2.5x stop   +Rs 59,979   Sharpe 1.61
    stop fired on 27 of 390 legs

### Over the months you asked about, it is not

| period | no stop | 2.5x stop | difference |
|---|---|---|---|
| Last week (23-29 Sep) | −39,140 | **−47,488** | **−8,348** |
| September | −19,861 | −20,197 | −336 |
| August | +38,860 | +37,902 | −959 |
| March (worst month) | −69,853 | −69,853 | **0** |
| **Full 8 months** | +45,470 | **+59,979** | **+14,510** |

The stop did **not** fix September, and it made last week materially worse.
Session by session it is easy to see why:

    28 Sep   -32,955  ->  -18,668    the stop worked, cut the worst day nearly in half
    24 Sep   -22,522  ->  -24,956    fired on a day that was going to be bad anyway
    29 Sep    +4,500  ->  -15,702    fired on a day that RECOVERED, locking in the loss

That last line is the cost of insurance, paid in public. On 29 September both
legs spiked past 2.5x intraday and then came back; without the stop the session
finished +Rs 4,500, with it -Rs 15,702.

### What this actually means

The stop's entire value lives in the rare session that would otherwise be
catastrophic. Across 158 sessions it fired on 52, helped on 26 and hurt on 26,
and 81% of its net benefit came from one day. Eight months contains enough of
those days for it to pay (+Rs 14,510); a single week usually does not, and this
particular week contained the bad kind.

The worst single session is unchanged at -Rs 48,097, because on 30 March both
legs topped out at 2.32x and 2.13x and never reached the trigger.

So: keep the stop, because over any sensible holding period it earns its place
and raises Sharpe from 1.27 to 1.61. Do not expect it to rescue a bad week. It
converts a small number of disasters into medium losses and charges a steady
premium for doing so, which is what insurance is.

## Preventing the September loss: what works, what does not (v43)

Eleven variants of the short straddle, all on real NSE bhavcopy prices, all with
the same point-in-time discipline (strike from the PRIOR close, liquidity from
the PRIOR session).

### The sweep

| variant | total | Sharpe | worst day | August | September |
|---|---|---|---|---|---|
| base (current) | +26,690 | 1.10 | −48,097 | +38,860 | −19,861 |
| **stop at 2.5x** | **+66,719** | **1.76** | −48,097 | +39,031 | −14,253 |
| stop at 2.0x | +58,533 | 1.45 | −43,786 | +25,108 | **+1,789** |
| stop at 3.0x | +38,633 | 1.32 | −48,097 | +36,929 | −17,494 |
| strangle 1% OTM | +30,520 | 1.22 | −47,614 | +26,936 | −16,316 |
| strangle 2% OTM | +5,080 | 0.53 | −25,658 | +7,894 | −7,687 |
| **iron condor 2% wings** | **−25,190** | 0.34 | −25,282 | +24,726 | −18,174 |
| iron condor 3% wings | −12,582 | 0.64 | −35,291 | +31,148 | −24,424 |
| skip after a big move | −18,738 | −0.09 | −32,955 | +40,829 | −35,598 |

**The iron condor loses money.** It was suggested twice in this project as the
obvious fix; the wings cost more than the protection is worth (−25,190 against
+26,690 for doing nothing). The "skip the day after a big move" filter is worse
than useless -- it sits out the calm recovery days and keeps the bad ones.

### The stop survives honest testing

Picked on the first half of the history, applied blind to the second:

    no stop    train +0.96 Sharpe   test +32,805 at 1.32
    2.5x stop  train +1.47 Sharpe   test +51,486 at 2.27

But the multiple matters more than it looks, because a stop is a market order
into a book that is already moving. At worse fills:

| stop | at trigger | 5% worse | 10% worse |
|---|---|---|---|
| 2.0x | +58,533 | **+1,547** | **−55,439** |
| **2.5x** | +66,719 | **+42,266** | **+17,813** |
| 3.0x | +38,633 | +32,207 | +6,503 |

2.0x fixes September on paper and collapses the moment fills slip, because a
tighter stop fires three times as often (62 sessions against 28). **2.5x is the
defensible setting**: it still earns +42,266 even if every stop fills 5% worse.

### What the stop does NOT fix

**March lost Rs 74,938, and every single variant left it untouched.** The worst
session in the whole sample, 30 March, shows why:

    BANKNIFTY 52300 PE   open 950.00   high 2,200.00 = 2.32x   close 2,037.10
    NIFTY     22800 PE   open 242.00   high   514.65 = 2.13x   close   477.50

Both puts more than doubled and neither reached 2.5x, so nothing fired and the
day cost Rs 48,097. A per-leg stop cannot see a position where both legs bleed
moderately at once.

And the stop is not free: across 158 sessions it fired on 52, helping on 26 and
hurting on 26, with **81% of its net benefit coming from a single session**. It
is insurance -- small regular cost, rare large payout -- not an edge.

### A position-level stop could not be tested

The obvious answer to 30 March is to stop on the PAIR rather than each leg. It
cannot be measured here: the bhavcopy gives each contract's daily high but not
when it happened, so `call_high + put_high` overstates what the pair ever traded
at together. Under that proxy every setting loses badly (−23,337 to −525,036)
and fires on up to 101 of 158 sessions, which says the proxy is broken, not that
the idea is. Testing it properly needs intraday option prices -- a broker feed,
not the end-of-day file.

### Recommendation

Add a **2.5x per-leg stop**. Measured: Sharpe 1.10 to 1.76, total +26,690 to
+66,719, and it holds out of sample and under 10% slippage. It improves
September from −19,861 to −14,253 rather than curing it, and does nothing for a
March. Anyone trading this should size for a −Rs 48,000 session, because that one
is still in the distribution.

## Making the loading bar actually visible (v42)

Two problems with the first version, both found by looking rather than assuming.

**It was too thin.** 3px against a dark page is invisible to anyone who does not
already know it is there. It is now 8px on a solid track with a bottom rule, a
triple glow, a bright spark riding the leading edge, and a much larger stage pill
with a pulsing dot and the percentage in mono type.

**It was painting behind the page.** `elementFromPoint(400, 4)` returned
`DIV.hero`, not the bar -- despite `position: fixed`, `top: 0` and `z-index:
9999`, all of which measured correctly.

The cause was a rule added in the FX layer:

    .stApp > * { position: relative; z-index: 1; }

which exists so content sits above the aurora. It also gives every Streamlit
container its own stacking context, and inside one of those a child's z-index
only ranks it against its siblings -- a later container still paints over it, no
matter how large the number. A `:has()` selector to lift the right container was
tried and did not match.

The fix that works is re-parenting: `boot_fx` now moves any `.loadwrap` it finds
to `document.body`, so the bar sits outside every Streamlit stacking context. It
re-runs on the same MutationObserver that drives the count-up numbers, because
Streamlit replaces the node on each update. Verified: parent is `BODY`, and the
bar renders above the page with its stage pill.

## A loading bar that fills as the analysis lands (v41)

Streamlit renders top to bottom, so the page arrives in stages -- price data,
news, the universe model, macro, the forecast engine (the slow one), the research
analyst. Until now the only signal was a spinner appearing and disappearing,
which says "something is happening" but never "how much is left".

There is now a 3px gradient bar pinned to the top of the viewport that advances
at each milestone, with the stage named beside it:

     6%  Starting up
    18%  Reading the news
    30%  Asking the universe model
    38%  Loading market & macro data
    48%  Training 5 forecast horizons
    78%  Scoring the model
    88%  Running the research analyst
   100%  Done -- flashes, then fades out

The bar itself runs a five-colour gradient flowing left to right on a 2.2-second
loop with a double glow, and the width transition is eased over 0.55s so each
jump glides rather than snaps. The stage pill is glass with a blurred backdrop.

Built on the same `st.empty()` slot pattern as the ticker tape and the scorecard
-- one placeholder, rewritten at each milestone -- so it costs nothing and needs
no JavaScript.

Verified in the browser mid-load: `position: fixed`, `top: 0`, `z-index: 9999`,
1280x38px, no ancestor creating a containing block that would trap it. Caught at
48% with the count-up numbers still climbing alongside it.

## The UI spec, audited against what already existed (v40)

A ten-point UI specification arrived. Rather than implement it blind, it was
checked against the codebase first -- and eight of the ten points were already
built, several of them months ago.

| spec item | status |
|---|---|
| Dark slate palette | already done (`#0b0e14` / `#151a23`) |
| Glassmorphic cards, hover translateY + shadow | already done (v38/v39) |
| Typography hierarchy, dimmed secondary text | already done (`#9aa4b2` / `#6b7483`) |
| Tabbed workspace | already done -- 7 main tabs plus 3 sub-tab groups |
| All controls in the sidebar | already done -- 22 sidebar bindings |
| Matplotlib/Seaborn to Plotly | **already Plotly** -- 13 figures, and `grep` finds zero matplotlib or seaborn anywhere |
| Transparent chart backgrounds | already done -- `paper_bgcolor`/`plot_bgcolor` are `rgba(0,0,0,0)` in the shared template |
| Range slider under candlesticks | already done -- 3 of them |
| Animated pulsing status badge | already done -- `.dot.live` |
| **Top banner with Win Rate / Sharpe** | **genuinely missing** |

The spec reads as though written for a generic Streamlit dashboard rather than
this one; the Matplotlib-to-Plotly conversion is the giveaway, since there has
never been a Matplotlib chart in this project.

### The one real gap, now closed

Win Rate and Sharpe Ratio specifically cannot be shown, because nothing here runs
a trading strategy -- there is no equity curve to take a Sharpe of, and inventing
one would be exactly the kind of decorative number this project has spent its
whole life removing.

What does exist is the model's own held-out quality, and it was buried 2,600
lines down inside a sub-tab. It is now a **Model scorecard** band at the top:

    Model skill vs "no change"   +0.19%   at 1 month · best +0.27% at 10d
    Horizons with an edge        4 / 5    beat the naive forecast out-of-sample
    Signal strength (alpha)      0.10     0 = no view trusted, 1 = fully trusted
    Data groups used             1 / 15   families that survived selection
    1-month uncertainty          ±9.3%    80% of outcomes land inside this

Every one is a number the engine already computed on held-out folds. It uses the
`st.empty()` slot pattern already in the file: the container is reserved near the
top and filled once training finishes, so it reads as a banner without forcing
the engine to train earlier than it does.

## UI v2: turned up until it is unmistakable (v39)

The first visual pass was too polite -- a 46px blur at 0.2 opacity reads as
"slightly nicer dark theme". This is the loud version.

### Visible now

**Aurora at ~3x the intensity.** Five saturated colour fields (blue, violet,
teal, pink, amber) at 0.55 opacity, blurred 52px and saturated 190%, drifting on
a 22-second loop with a 40-second hue rotation underneath. It is the first thing
you see.

**A scan line** sweeps top to bottom every 9 seconds in screen blend mode.

**Neon rims that breathe.** Every KPI card and metric pulses between a blue and a
violet glow on a 4.5-second cycle, staggered so the grid ripples rather than
flashing in unison. Hover lifts a card 10px, scales it 4%, and fires a 46px glow
plus a light sweep across it.

**The hero** carries a 2px conic-gradient rim through five colours spinning every
5 seconds, with the title in animated gradient text that glows in and out.

**Section headers** get a 3px underline that draws itself and then flows colour
through, and the numbered kicker floats.

### JavaScript, which CSS could not do

`st.markdown` strips `<script>`, but `components.v1.html` renders a real iframe,
and from inside it `window.parent.document` reaches the app. `ui.boot_fx()` uses
that for two things:

* **Count-up numbers** -- every KPI, metric and hero price ticks from zero to its
  value over 900ms on a cubic ease-out, firing when it scrolls into view. The
  prefix and suffix are preserved, so `Rs 1,23,456`, `$578.42`, `+5.24%` and
  `20.02M` all animate without losing their formatting, then snap back to the
  exact original string so nothing is ever left rounded.
* **A cursor glow** -- a 460px radial light follows the pointer.

Both re-arm through a MutationObserver, because Streamlit replaces DOM nodes on
every rerun. The iframe is 0px tall and adds nothing to the layout.

### Measured, not assumed

Frame rate with everything running: **59.7 fps**. Animating `filter` on a
full-screen blurred layer is the one genuinely expensive thing here, so it was
worth checking rather than hoping -- it is fine.

All of it switches off under `prefers-reduced-motion`.

**Remember:** `ui_theme` is imported once, so theme edits need a server restart,
not just a rerun. Then hard-refresh the browser (Ctrl+Shift+R) so it does not
serve cached CSS.

## UI: motion, glass and glow (v38)

A visual pass over the whole dashboard. Everything is CSS -- Streamlit strips
<script> from markdown -- so it all runs on the compositor (transform, opacity,
filter, gradients) and never touches layout, which is what keeps a page with this
much motion smooth. 16 keyframe animations, ~11k characters of new styling.

**Background.** A single fixed aurora layer behind the app: four coloured blobs
blurred at 46px, drifting on a 26-second loop, with a faint grid panning beneath
it under a radial mask. One element, low opacity, GPU-composited.

**Depth.** Cards, hero, TL;DR and live panels are glass -- `backdrop-filter:
blur(12px) saturate(130%)` over a translucent surface, so the aurora shows
through.

**Hero.** A conic-gradient border rotating through blue, violet, teal and pink on
a 7-second cycle, with the title and price in animated gradient text.

**Cards.** KPI tiles stagger in on a 50ms cascade, lift 6px and scale 1.8% on
hover, gain a blue rim and a 34px glow, and a light sweep crosses them. Charts,
tables and metrics fade up on entry and lift on hover. Buttons lift, brighten and
sweep. Tabs get a glowing sliding indicator. The sidebar logo floats on a
4.2-second loop inside a rotating conic halo, and the brand name shimmers.

**Details.** Custom scrollbar, glowing focus rings on inputs, a slider thumb that
scales on hover, animated underlines that draw themselves under section titles.

All of it is disabled under `prefers-reduced-motion`.

### Two things fixed along the way

**The sweep was a pseudo-element, and that clipped the data.** The shimmer needed
`overflow: hidden` on the card, which activated the `text-overflow: ellipsis`
already on `.kpi-value` -- turning the analyst target into "$578. ...". The sweep
is now an animated background layer instead, so nothing needs clipping.

**KPI values were truncating anyway.** At six columns each tile is only 87px
wide, and `nowrap + ellipsis` was cutting "$578.42" and "+5.24%". Values now use
`clamp(1.0rem, 1.6vw, 1.35rem)` and wrap instead of clipping -- a second line
beats an ellipsis on a price.

Note for anyone editing the theme: `ui_theme.CSS` is a module-level constant, so
Streamlit reruns do NOT pick up changes to it. The server must be restarted.

## Zerodha Kite Connect for live bid/ask (v36)

The free NSE live feed carries lastPrice and nothing else, so the spread -- the
thing that decides your actual fill -- was invisible. Against an edge of about
+0.12% of margin per session that gap matters. `kite_client.py` closes it.

### What it does

Read-only Kite Connect over plain HTTPS, no new dependency (the official
`kiteconnect` package is optional):

    Kite.quote(symbols)        five-level depth: bid, ask, sizes, volume, OI
    depth_frame(quotes)        flattens it to bid / ask / mid / spread / spread%
    realistic_straddle_fill()  what a SELLER actually receives (the bid) against
                               the optimistic lastPrice number, and the gap

On the real NIFTY 22600 straddle quoted earlier:

    sell at LTP (optimistic)  274.85 pts
    sell at BID (realistic)   274.35 pts
    spread cost                 0.50 pts = Rs 38 per 75-lot

Small in isolation, and roughly 13% of the strategy's Rs 288 average session.
That is exactly the kind of number that decides whether a thin edge survives.

### Credentials: yours, never in a chat and never in the code

The module never prompts for a secret and none is stored in it. Put them in
environment variables (`KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`)
or in `kite_credentials.json` beside the module -- which is now in `.gitignore`,
because it is a key to a real brokerage account.

    python kite_client.py status      what is configured, and whether it works
    python kite_client.py login       prints the Zerodha login URL
    python kite_client.py token XXX   exchanges the request_token, saves it
    python kite_client.py quote SYM   live depth for a contract

You log in yourself in Zerodha's own browser window. The code never sees your
password and never places an order -- it only reads quotes.

### What you have to do, which cannot be automated

1. Subscribe to Kite Connect at developers.kite.trade (about Rs 2,000/month).
   A Zerodha trading account by itself is not enough.
2. Create an app, note the api_key and api_secret.
3. Mint an access token once per day -- they expire around 6 a.m. That is
   Zerodha's design, not a limitation of this code.

### Verified without an account

`test_kite_client.py` runs on recorded payloads in Kite's exact shape and
passes 15 checks: top-of-book selection (not a deeper level), mid, spread and
spread%, seller-receives-the-bid arithmetic, missing depth returning None
instead of a guessed price, the SHA256 checksum, refusal to act when
unauthenticated, and no literal secret anywhere in the source.

One of those checks initially failed and the TEST was wrong, not the code: it
asserted mid <= LTP, but mid legitimately exceeds LTP whenever the last trade
printed below the midpoint. The invariant that actually holds is bid <= mid <=
ask, and that is what it now checks.

So the integration is known-good before anyone pays for a subscription.

## Does it work in real time? (v35)

Yes for monitoring and strike selection, no for knowing your fill. Both halves
were tested rather than assumed.

### The bhavcopy is end-of-day only

Checked at 12:03 on 30 Sep 2026 with the market open:

    2026-09-30 (TODAY)        not published
    2026-09-29 (1 day ago)    AVAILABLE, 37,922 rows
    2026-09-28 (2 days ago)   AVAILABLE, 37,484 rows

So the eight-month backtest can be run on history, but nothing can be traded
from that file during a session.

### There IS a live feed, and it works

NSE's option-chain API returns an empty body, and `quote-derivative` 404s. But
this one answers:

    https://www.nseindia.com/api/liveEquity-derivatives?index=nse50_opt

1,577 NIFTY contracts, stamped to the minute, with lastPrice, open, high, low,
volume, open interest and the live underlying. `nse_live_options.py` wraps it.
Live output at 15:22:15 on 30 Sep:

    spot 22,616.60   strike 22,600   expiry 06 Oct   lot 75

    cp   open    last     high     low       volume       OI
    CE  261.00  159.65   311.70  155.05   58,338,735   67,527
    PE   75.00  115.20   134.35   54.50  192,426,390   98,746

    straddle at the open   336.00 pts
    straddle now           274.85 pts
    P&L if sold at open    Rs +4,466  on ~Rs 407,099 margin  (+1.10%)

### The limitation that actually matters

**The feed carries lastPrice but no bid and no ask.** So the spread -- which
decides the price you are really filled at -- is invisible. The backtest assumed
a sell at the session's open; a live order is filled at the bid, which is worse.
The measured edge is about +0.12% of margin per session, thin enough that a
couple of points of spread on a straddle is a meaningful share of it. Every live
P&L this module prints is an **upper bound**, not an achievable price.

Also, only `nse50_opt` responds. The BANKNIFTY and all-F&O variants return
HTTP 500, so live coverage is NIFTY only.

### What this means practically

    live strike selection        yes
    live position monitoring     yes
    live P&L (optimistic)        yes
    realistic fill estimate      NO -- needs bid/ask
    automated order placement    NO -- needs a broker API

To close the last gap you need a broker feed (Zerodha Kite, Dhan, Upstox all
publish bid/ask over websocket, roughly free with an account). That is the one
missing input between this and a system you could actually trade.

## Real exchange option data, and the first credible positive result (v34)

Told to fetch data from wherever. NSE's option-chain API is closed, but the
exchange's **daily F&O bhavcopy** is not: one row per contract per session with
real open/high/low/close, settlement, open interest and volume.

    https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip

`nse_fo_data.py` downloads and caches it. 163 sessions, 466,337 option rows,
Feb-Sep 2026. Every options conclusion before this used Black-Scholes prices
invented from realised vol; they can now be replaced with what actually traded.

### Buying options: the claim confirmed, with real prices

8,552 liquid contracts, <=10 days to expiry, open to close:

    gained >100%   2.8% of contracts      biggest  +1,709%
    gained  >50%   5.5%
    lost   >50%   15.1% of contracts      biggest    -100%
    MEDIAN day  -17.50%      MEAN day  -12.91%

Double-digit and triple-digit days are real and frequent. Losses over 50%
outnumber gains over 100% by five to one, and the average contract loses 12.91%
a day. The real data is harsher than the Black-Scholes estimate of -8.83%.

### Selling options: three look-ahead leaks, each one spectacular

The other side of that trade looked extraordinary, and was wrong three times.

**Leak 1 - selecting on that day's volume.** Volume is known at the close, not
the open. Seller P&L by volume quintile: +0.48, -12.62, -8.18, +7.29, +12.26
points. Sorting on it hands you a 25x spread for free.

**Leak 2 - the window was the calmest in years.** NIFTY realised 8.4% annualised
over Aug-Sep against 15.9% long-run, a ratio of 0.53x. Biggest daily move 1.64%,
against 12.98% in twelve years.

**Leak 3 - the worst.** `UndrlygPric` in the bhavcopy is the CLOSING spot,
verified at 0.0000% mean difference against the NIFTY close. Selecting "ATM"
with it picks the strike that ENDED at the money -- exactly the strike that
decayed most that day.

    with leaks 1+3 present    Sharpe 17.51,  96% of sessions positive
    all three fixed           Sharpe  1.27,  68% of sessions positive

A Sharpe of 17 is not a discovery, it is a receipt for a bug.

### What survives

Short the ATM call and put on NIFTY and BANKNIFTY at the open (strike from the
PRIOR close, liquidity from the PRIOR session, <=7 DTE), cover at the close.
Margin 12% of notional, Rs 60 per lot round trip. 158 sessions:

    total P&L            Rs +45,470
    mean per session     Rs +288   (+0.12% of margin)
    monthly              approx +2.4% of margin
    winning sessions     68%
    Sharpe               1.27
    worst session        Rs -48,097  (-5.7%)
    max drawdown         Rs -94,210

| month | sessions | P&L |
|---|---|---|
| Feb | 18 | −1,329 |
| Mar | 18 | **−69,853** |
| Apr | 18 | +45,754 |
| May | 19 | +17,922 |
| Jun | 21 | +21,017 |
| Jul | 23 | +12,960 |
| Aug | 21 | +38,861 |
| Sep | 20 | −19,861 |

**This is the first credibly positive result in the project.** It is also honest
about what it is: the maximum drawdown is twice the total profit, two of eight
months lost money, March alone gave back more than a year's worth of grinding,
and eight months of below-average volatility is not long enough to have met a
real crash. Short volatility pays small and often and takes it back rarely and
violently.

About +2.4% a month on margin, not 10%. Sharpe 1.27 is a genuinely good number --
for reference, Renaissance Medallion, the best record that exists, is around 2.5.
Anything far above that in a backtest is a bug, as this file demonstrates three
times over.

## "Options move 100% in a day" — correct, and here is what it costs (v33)

Pushed back on the conclusion with a specific, checkable claim: options routinely
move 100%+ in a day. That is true, and it deserved measuring rather than arguing.

Yahoo's live `percentChange` is all zeros on a closed-market snapshot, so option
prices were reconstructed from REAL NSE underlying moves over 12 years: buy an
out-of-the-money weekly call at the open, sell at the close, 37,416 sessions
across 12 symbols. OTM deliberately, because that is where the big moves live.

### The claim is confirmed

| strike | gains >100% | gains >50% | loses >50% | loses >80% | median day | mean day |
|---|---|---|---|---|---|---|
| ATM | 5.3% | 13.7% | 17.3% | 2.8% | −14.0% | −2.7% |
| +1% OTM | 7.0% | 15.1% | 22.7% | 4.6% | −19.0% | −2.3% |
| +2% OTM | 8.4% | 15.8% | 28.7% | 7.3% | −24.8% | −1.1% |
| **+3% OTM** | **9.3%** | 16.0% | 34.8% | 10.7% | −31.2% | **+2.0%** |

A +3% OTM call more than doubles on **9.3% of sessions** — about once every
eleven trading days. The single best day in the sample was **+10,460%**. Its
average day is **positive, +2.0%**.

### And buying it every day still goes to zero

    Rs 100,000, full size, 2,000 sessions  ->  Rs 0, wiped out

That is not a contradiction. The median day is −31.2%. A handful of +1,000% days
lift the ARITHMETIC mean, but capital compounds MULTIPLICATIVELY, and you cannot
survive to collect the rare winners. Positive average, negative geometric growth.

Position sizing does not rescue it:

| fraction of capital per trade | ending value |
|---|---|
| 100% / 25% / 10% | Rs 0 — wiped out |
| 5% | Rs 3 |
| 2% | Rs 3,710 |
| 1% | Rs 22,927 |

Even at 1% per trade, Rs 100,000 becomes Rs 22,927.

### The part that settles it: what you actually pay

The +2.0% average came from pricing options at REALISED volatility. The market
charges IMPLIED volatility, which measured ~3.4 points higher. Repricing at what
is actually quoted:

| you pay | mean day | best outcome at any sizing |
|---|---|---|
| realised vol, no spread | +0.98% | Rs 138,126 |
| realised vol + 1% spread | +0.98% | Rs 65,426 |
| IV = realised +2 pts | **−6.11%** | Rs 13,359 |
| **IV = realised +3.4 pts (measured)** | **−8.83%** | Rs 6,953 |
| IV = realised +5 pts | −10.88% | Rs 2,912 |

The positive average existed only while the options were priced too cheaply. At
real market prices every trade loses **8.83% on average**.

This closes the loop with the variance risk premium measured earlier: implied vol
sits ~3.4 points above realised because sellers demand payment for carrying the
risk. That premium is the seller's edge — which makes it precisely the buyer's
cost. The same number that makes selling options mildly profitable is the number
that makes buying them reliably unprofitable.

**So the 100% days are real, frequent, and not the point.** They are financed by
the 31% median loss on every other day, and by paying a premium above fair value
for the privilege.

## The overnight effect: the best result in the project, and it was false (v32)

Asked to search everything available for something that could make the project
profitable. The strongest lead found was the overnight/intraday decomposition
(Lou, Polk & Skouras 2019; NY Fed "Overnight Drift"): published work finds nearly
all equity return accrues overnight while the intraday session drifts near zero.
Every strategy here traded intraday, so this would have explained everything.

On NSE daily bars it looked spectacular:

    overnight  +39.9%/yr   positive in 29/29 symbols   Sharpe 1.84
    intraday   -17.6%/yr   positive in  1/29 symbols   Sharpe -0.59
    NIFTY overnight +31.1%/yr, Sharpe 2.80, against +11.0% buy & hold

NIFTY futures cost 0.0348% round trip, so even 252 round trips a year (~13.9% of
capital) would have left roughly **+17%/yr net** -- comfortably the best result
anywhere in this project.

### Then the validation

The decomposition assumes the daily Open is the first traded price. Checked
against the first 5-minute bar of the same session:

| symbol | sessions | mean abs diff | Open == first trade |
|---|---|---|---|
| HDFCBANK.NS | 58 | 0.128% | 38% |
| RELIANCE.NS | 58 | 0.082% | 43% |
| TCS.NS | 58 | 0.145% | 29% |
| INFY.NS | 58 | 0.155% | 28% |

Yahoo's daily Open for NSE symbols is **not** the opening trade. It disagrees
with the tape on about two sessions in three, by ~0.11% on average -- and
0.11% x 252 is ~25%/yr, which is the size of the entire "effect".

Recomputed from intraday bars only, with no daily Open anywhere in it, it
collapses or reverses: HDFCBANK -28.6%/yr overnight, RELIANCE -15.0%, TCS +12.1%.
The US indices, whose Open field is trustworthy, show no such split
(^GSPC +7.4% overnight against +4.3% intraday).

**A Sharpe-2.8 strategy that exists only in one vendor's Open column is a bug in
the data, not an edge in the market.** Not shipped. `exp_overnight_effect.py` is
kept in the repo so it cannot be rediscovered and believed.

## "Options can easily give 10-15% in a day" (v31)

Asked to tune the model for 10-15% a day through options, at low risk, on the
grounds that options move that much routinely. The first part of that is true and
worth confirming properly; the rest does not follow. `exp_option_leverage.py`
takes the SAME opening-range-breakout trades the intraday work produced, 452 of
them over 58 sessions, and runs them through Black-Scholes ATM option economics:
priced at entry and exit on each symbol's own realised vol, theta over the hold,
bought at the ask and sold at the bid.

### The premise is correct

    median leverage           an ATM option moves 52.9x the underlying, in %
    trades gaining  > +10%    26.5%
    trades gaining  > +15%    23.7%
    best single trade         +148.6%

Double-digit days are not rare. They happen on a quarter of trades.

### The other side of the same coin

    trades losing   < -10%    59.7%
    trades losing   < -15%    49.1%
    worst single trade        -61.5%

More than twice as many trades lost 10% as gained it. That asymmetry is not bad
luck: the win rate is 34%, and theta plus the spread are charged on every trade
whether it works or not.

### Leverage multiplies the edge, and the edge is negative

    underlying, gross of costs   +0.074% per trade
    underlying, net of costs     -0.076% per trade
    the same trades as options   -0.812% per trade

A multiplier applied to a negative number stays negative, and gets bigger.
Compounding Rs 100,000 one position per signal:

| position size | ending value | max drawdown |
|---|---|---|
| 100% of capital | **Rs 852** | −99.2% |
| 25% of capital | Rs 11,661 | −93.5% |
| 10% of capital | Rs 56,435 | −61.9% |

Full sizing wiped out the account. The gentlest sizing still lost 44%.

### Why the target itself cannot exist

    10% a day for a year  ->  Rs 2,697,470,226,775,856
    15% a day for a year  ->  Rs 197,631,318,980,111,335,424

One lakh becomes more than the entire Indian equity market inside a year. That is
not a target to tune toward; it is the proof that the target cannot exist. Anyone
holding such a rate would own every listed company within months.

### Where the real options edge is: selling, not buying

Yahoo's `impliedVolatility` field is unusable (bid/ask all zero, IV median 0.0000
on a stale snapshot), so IV was backed out of traded prices with a bisection
solver instead:

| symbol | ATM IV | realised vol | IV − RV |
|---|---|---|---|
| SPY | 20.1% | 11.3% | **+8.8** |
| QQQ | 29.6% | 18.3% | **+11.3** |
| JPM | 27.5% | 16.9% | **+10.6** |
| KO | 21.2% | 13.7% | **+7.5** |
| MSFT | 33.0% | 45.5% | −12.5 |
| AAPL | 25.1% | 27.5% | −2.4 |

    mean IV 26.1%   mean realised 22.7%   gap +3.4 points, positive in 5 of 7

That gap is the **variance risk premium**, the one options edge with strong
published support, and it is paid to option SELLERS. It is the exact opposite of
buying calls and puts for direction. It is also where this project's single
genuine edge already lives: the HAR-RV volatility model forecasts realised vol
12.1% better than a trailing window, which is precisely the tool for deciding when
implied is rich.

It is worth perhaps 3-4 volatility points, it carries real tail risk that needs
defined-risk spreads to contain, and it is nothing like 10-15% a day. It is,
however, real, which none of the alternatives were.

## Intraday paper trade of last week (v30)

Asked to paper-trade intraday suggestions on last week and make the most profit
possible. "Most profit on last week" is trivial to fake -- try enough rules on
last week and one looks brilliant -- so the week was never allowed to choose
anything:

    TRAIN  53 sessions (09 Jul - 22 Sep), 5-minute bars, 8 NSE large caps.
           22 rule/parameter combinations scored here. Winner picked here.
    TEST   last week (23-29 Sep), traded once, blind, rule already frozen.

Rules tested: opening range breakout (3 lengths x 2 stops), intraday momentum
(Gao, Han, Li & Zhou 2018 -- first half-hour predicts last half-hour, 9
variants), VWAP reversion (3), gap fade and gap continuation (4).

### Training: not one rule was profitable

    0 of 22 rules had positive expectancy after costs

The best was ORB-15min at **-0.072%** per trade over 413 trades. The "winner" was
the least-bad loser, which means there was nothing to pick.

### Last week, out of sample

    rule          ORB 15min stop0.5 (frozen before the week)
    trades        39  (12 winners, 27 losers, 31% win rate)
    expectancy    -0.121% per trade
    P&L           -588 on 100,000  (-0.59%)
    training -0.072% -> test -0.121%   did not hold up

Day by day it ran 99,829 / 99,833 / 99,632 / 99,904 / 99,412. The best single
trade was RELIANCE short +1.21%; the worst AXISBANK short -0.82%.

### Why — and this is the sharpest result in the project

Separating the gross edge from the cost of capturing it:

| rule | GROSS edge | net at 0.15% |
|---|---|---|
| ORB 15min stop0.5 | **+0.078%** | −0.072% |
| ORB 30min stop0.5 | +0.071% | −0.079% |
| ORB 60min stop0.5 | +0.070% | −0.080% |
| Gap continuation >0.2% | +0.046% | −0.104% |

**8 of 22 rules have a positive gross edge.** The intraday patterns are real.
ORB-15min earns +0.078% per trade before costs, so it breaks even at a round trip
of 0.078%.

Statutory charges alone -- brokerage, STT, exchange fees, SEBI, stamp duty, GST --
come to **0.082%** on a Rs 100,000 intraday position with a discount broker. Before
a single paisa of slippage, the toll already exceeds the edge.

That is the whole story of this project in one comparison:

    intraday edge found   +0.078% per trade
    cheapest possible toll -0.082% per trade
    result                 -0.004% before you even slip on a spread

The patterns are not imaginary and the models are not broken. The edge is simply
thinner than the cost of collecting it, which is exactly what an efficient market
looks like from the retail side. Trading more often makes it worse, because every
extra round trip pays the toll again.

What survived everything tested, over five years out of sample: equal-weight buy
and hold, +10.4%/yr at Sharpe 0.75, with one trade.

## Where we were lagging, and what actually makes money (v29)

Asked to find what is wrong, fix it, and produce something profitable. Here is the
honest chain of findings.

### The fix that helped, and still lost

Three measured problems with the losing paper trade, each fixed:

1. **The target was decorative.** 1.5xATR sits 3.1% from the open; only 3.1% of
   sessions travel that far. It had a loss branch and no win branch. Removed --
   hold-to-close beat every bracket setting.
2. **Costs were wrong.** 0.30% is a DELIVERY round trip. Intraday is 0.082% in
   charges plus slippage = **0.15% all-in**. I had been over-charging every trade
   by roughly 2x.
3. **The model is anti-predictive when confident.** Expectancy falls monotonically
   with conviction: -0.109% in the weakest signal decile, **-0.621% in the
   strongest**, win rate 44.5% -> 32.5%. At zero cost the top decile still loses.

Fading high-conviction calls survived two independent out-of-sample periods
(+0.405% and +0.235% per trade, positive in both for 7 of 9 symbols), so v2 fades
the signal, trades only high-conviction days, and holds to the close.

    v1, 5 sessions    -2.37%   24 trades, 25% win    vs buy & hold -0.25%
    v2, 5 sessions     0.00%   stood aside on all 24 (nothing cleared the bar)
    v2, 12 weeks      -3.28%   46 trades, 46% win    vs buy & hold -6.27%

Better -- it beat buy-and-hold by 3 points over 12 weeks -- but still negative.

### The real diagnosis

The problem is not the features, the learner or the tuning. It is the QUESTION.
"Which way does this liquid large cap move tomorrow?" is the single most
competed-over question in markets, and every measurement here says it has no
answer from daily Yahoo bars.

So the documented, replicated equity anomalies were tested instead, on 30 NSE
large caps, split 60/40 in time. **Out-of-sample, Sep 2021 - Sep 2026:**

| strategy | annualised | Sharpe | max drawdown | rebalances |
|---|---|---|---|---|
| **Equal-weight buy & hold** | **+10.4%** | **0.75** | −17.1% | 1 |
| Momentum 12-1 (long-only) | +9.4% | 0.56 | −26.6% | 59 |
| Reversal 10d (long-only) | +7.5% | 0.48 | −22.1% | 125 |
| NIFTY 50 | +5.8% | 0.48 | −17.2% | 1 |
| Low volatility (long-only) | +4.8% | 0.38 | −15.8% | 59 |
| Momentum 12-1 (long/short) | −1.5% | 0.02 | −32.7% | 59 |
| Reversal 10d (long/short) | −6.6% | −0.27 | −36.4% | 125 |
| Same-day ML bracket (v1) | ~−70% | — | — | daily |

Two things fall out of that table.

**Nothing beat holding everything.** Momentum's +9.4% looks respectable until you
see equal-weight buy & hold at +10.4% with a much smaller drawdown and a better
Sharpe, achieved with one trade instead of 59. Every signal tested was a worse way
to own the same stocks.

**Long/short is where the truth is.** Long-only results are inflated by
survivorship -- the universe is today's large caps, so the failures are missing.
Long/short cancels most of that, and every long/short variant came out at or below
zero. That is the cleanest statement of the result: no tradeable alpha was found.

### What is shipped

`paper_trade_portfolio.py` paper-trades the approaches that survived, against the
index, with costs. Over the full out-of-sample period on Rs 100,000:

    Equal-weight buy & hold   +63,772   (+10.4%/yr, Sharpe 0.75)
    Momentum 12-1             +56,417   (+9.4%/yr,  Sharpe 0.56)
    NIFTY 50                  +32,843   (+5.8%/yr,  Sharpe 0.48)

Over the last 12 months alone all three lost (-6.9%, -10.0%, -7.9%) because the
market fell. That is the point of the multi-year window, and why a one-week paper
trade cannot answer this question.

The defensible conclusion from everything measured across this project: this
system's value is in **risk, timing and expectation management** -- position
sizing from ATR, honest probabilities, calibrated uncertainty bands, volatility
forecasting (HAR-RV, the one genuine +12.1% edge found) -- and not in predicting
direction. The forecasting machinery is worth keeping for what it says about
UNCERTAINTY. It is not worth trading on for DIRECTION, and the app now says so
wherever it would otherwise imply otherwise.

## A one-week paper trade (v28)

Asked for a week of paper trading and what it earned. A forward test takes a week
of wall-clock time, so this is a **replay** of the last five sessions under
point-in-time discipline: the model is trained only on data dated strictly before
the week, then trades those five days blind. `paper_trade.py`, re-runnable.

Six NSE symbols, equal weight, Rs 100,000, 0.30% round trip.

    trades                24   (6 winners, 18 losers, 25% win rate)
    exits                 target 0  stopped 7  closed out 17
    strategy return       -2.37%   ->  -2,367
    buy & hold same days  -0.25%   ->    -247
    best / worst trade    +1.44% / -2.51%

**It lost Rs 2,367, and it lost ten times more than doing nothing.**

### One week proves nothing, so the week is shown inside its distribution

    the same rules over all history: 3,756 weeks
      mean week    -1.78%      weeks that made money: 24%
      median week  -1.79%      std dev of a week: 2.93%
      best  +19.69%            worst -23.98%

    this week: -2.37%, the 41st percentile

A week's spread is +/-2.93% against a long-run mean of -1.78%. This was an
ordinary week for a losing strategy, not an unlucky one. The script prints how
many weeks would be needed to separate the effect from noise, which for an effect
this large is about 10 -- and the sign is already clear.

### What the log exposed: the target was decorative

Target hit **0 times in 24 trades**. Not bad luck. At the default 1.5xATR the
target sits 3.1% from the open, and only 3.1% of sessions travel that far:

| target | = % of price | target hit | stopped | drifted to close | bracket | hold to close |
|---|---|---|---|---|---|---|
| 0.3xATR | 0.62% | 52.9% | 18.7% | 28.4% | -0.426% | -0.355% |
| 0.8xATR | 1.65% | 17.0% | 18.7% | 64.3% | -0.374% | -0.355% |
| 1.5xATR | 3.10% | 3.1% | 18.7% | 78.2% | -0.356% | -0.355% |
| 2.0xATR | 4.13% | 1.1% | 18.7% | 80.2% | -0.353% | -0.355% |

The bracket had a loss branch and effectively no win branch. And **no target
setting beats simply holding to the close**: tighten it and you cap the winners
while keeping the whole stop loss; widen it and it is never reached. The stop
earns its place as risk control. The target does not.

The plan tab now states how often the chosen target has actually been reached and
says so plainly when the answer makes it decorative.

## Target-move finder, same-day plan, and the number nobody wants (v27)

### "I want 1-2%. When do I get it?"

The window scan works on daily candles, so the shortest thing it could ever suggest
was a multi-day hold. A 1-2% move is usually an intraday event, so it was the wrong
model for the question.

A daily candle already contains the session's High and Low, so "did this reach +1%
above the open at some point today" is answerable from daily bars over the FULL
history -- 3,148 sessions for HDFCBANK.NS instead of the 58 that Yahoo's 15-minute
feed allows. Hit rates come from daily bars; intraday bars are used only for WHEN in
the session the move lands, and that smaller sample is labelled as such. Cross-checked
on the same 58 sessions: daily-derived 51% / 36% at +0.5% / +1.0% against
15-minute-derived 52% / 28%.

HDFCBANK.NS, measured:

| Target | Same-day odds (long) | Typical time | Within 3 days | Typical wait | Watch |
|---|---|---|---|---|---|
| 0.5% | 64% | 22 min | 85% | 1 day | 5m |
| 1.0% | 33% | 2.1 h | 67% | 2 days | 30m |
| 1.5% | 17% | 4.6 h | 53% | 3 days | 1h |
| 2.0% | 8% | — | 41% | 4 days | 1h |

So 1-2% is a **2-4 day** question, not a two-month one.

One correction to the premise it was built on. "A stock moves 1-2% in a day" is true
of its RANGE -- 63% of sessions touch +/-1% from the open -- and false of what you
can capture, because you cannot buy the low and sell the high. Going long from the
open, +1% arrives on **33%** of sessions. Both numbers are shown; the directional one
is the one the page plans with.

### Same-day plan: entry, stop, target, CALL/PUT

Distances are set in ATR rather than round numbers, so they scale with how much the
symbol actually moves: 1% means something different on a quiet large cap and a
volatile small cap, 0.8xATR means the same thing on both.

**Option chains:** Yahoo serves them for US symbols and not for Indian ones --
HDFCBANK.NS, RELIANCE.NS and NIFTY all return zero expiries, and NSE's own
option-chain endpoint is closed to public scraping (tested, 200-with-empty-body and
404). So US symbols get real strikes, bid/ask and break-even; Indian symbols get the
underlying levels plus a stated strike-selection rule and **no premium numbers at
all**, because inventing them would be lying.

### The number nobody wants: what does the model guarantee?

Nothing. There is no function in this project that returns a guaranteed return,
deliberately. What it reports instead is the measured record of the exact rule, and
that record is negative.

Every stop/target combination on HDFCBANK.NS, replayed over 3,134 sessions with a
0.30% round trip:

    best of 16 settings   -0.30% per trade
    worst                 -0.38% per trade
    win rate              30-40%, average win +0.8%, average loss -0.9%

Set the cost slider to zero and the expectancy becomes **-0.004%** with a 48.9% win
rate. That is the finding: **the entry has no edge at all, and the round trip is the
entire loss.**

Nor does the model's own direction signal rescue it. Walk-forward across 6 symbols,
taking the trade only when the model points that way:

| | signal | blind (always long) |
|---|---|---|
| expectancy | −0.336%/trade | −0.281%/trade |
| profitable after costs | **0 of 6 symbols** | 0 of 6 |
| beats the other | 2 of 6 | 4 of 6 |

The signal version is *worse* on average. So the plan tab shows its own record
**above** the levels, in red, and says the levels are risk management for a trade
you have already decided to take rather than a reason to take one. A screen showing
entry/stop/target without that record would be the most dishonest thing in this
project.

The backtest also states its own weakness: a daily candle records the High and Low
but not their order, so sessions that touched both target and stop are counted as
**losses**. That convention affects 0.1-5% of sessions depending on settings, and
`test_options_plan.py` verifies that flipping it to optimistic would only RAISE the
number -- so what is reported is a floor.

### Why Random Forest, if it picks features randomly?

The premise is reasonable and the measurement says the opposite of what it predicts.

**It is not random at run time.** Five identical refits on the same data produce
bit-identical predictions (spread 6.9e-18). The randomness is in how each tree is
BUILT -- each split sees a random subset of features -- and its purpose is to stop
300 trees from being 300 copies of the same tree. Averaging decorrelated trees is
what removes variance; it does not add noise to the output. Changing the seed moves
the forecast by 0.23% against a target whose own spread is 6.03%, i.e. **3.8% of the
signal scale.**

**It is the strongest member, not the weakest.** Held-out skill against a no-change
forecast, 8 symbols x 2 horizons:

| member | mean | median | worst |
|---|---|---|---|
| Random Forest alone | **−7.62** | −6.08 | −22.98 |
| XGBoost alone | −14.19 | −13.28 | −43.89 |
| Ridge alone | −26.00 | −20.51 | −68.88 |
| Ridge+RF+XGB | −11.75 | −9.17 | −42.64 |
| Ridge+XGB (no RF) | −16.19 | −13.44 | −54.52 |

Removing RF costs 4.44 points of skill, and the blend containing it wins in
**16 of 16** cases.

Read the signs on that table: every raw member is NEGATIVE. Ungated, all three lose
to simply predicting "no change" -- which is exactly why the engine shrinks each
model toward no-change by a factor fitted on out-of-fold data and forces it to zero
unless it beats naive in at least 2 of 3 folds. The value of the shipped system is
not in the raw models. It is in knowing when not to trust them.

## The buy/sell window was recommending the slider (v26)

### "2 mahine rakhne bol rha tha, bas 1.4% ke liye"

Two months of your money tied up for 1.4%. The complaint was right, and the cause
was worse than slow returns — **the holding period was never a finding at all.**

The old scan did exactly this:

    buy  = argmin(forecast path)
    sell = argmax(path after the buy)

The forecast path rises almost monotonically — a small drift, no oscillation — so
`argmin` is day 1 and `argmax` is the last day, every single time. Proof on
HDFCBANK.NS, changing nothing but the slider:

| slider | buy date | hold | "gain" |
|---|---|---|---|
| 5 | 30 Sep | 6 d | 0.14% |
| 30 | 30 Sep | 41 d | 0.64% |
| 60 | 30 Sep | 83 d | 2.10% |
| 90 | 30 Sep | 125 d | 3.23% |

Same buy date in all four. **The "recommended" holding period was just the slider
value.** And the gain it advertised ignored the two things that decide whether a
trade is worth doing at all:

* **costs** — a round trip in Indian equities runs ~0.30% (brokerage, STT,
  exchange fees, slippage). That 0.14% "gain" was a loss.
* **uncertainty** — at the 83-day sell date the model's own 80% band was **23.8%
  wide**. A 2.15% expected gain sitting inside that is a coin flip, not a plan.

### What it does now

Every (buy, sell) pair is scored on what actually matters:

    net        = gross - round-trip cost
    sigma_ij   = sqrt(sigma_j^2 - sigma_i^2)     the uncertainty of the HOLD,
                                                 not of the price level
    p(profit)  = Phi(net / sigma_ij)
    annualised = (1 + net) ^ (365/days) - 1

The headline is the **shortest** hold that clears both bars, because the question
was how to stop waiting two months for 1.4%. The best-odds and best-per-year
windows sit beside it, along with plain buy-and-hold, so the trade-off is visible
instead of hidden. A frontier table shows what each holding length can actually buy.
Both bars are yours to set — the round-trip cost and the minimum chance of profit
are sliders.

The same HDFCBANK.NS, after:

| slider | verdict | hold | window |
|---|---|---|---|
| 30 | **no trade** | — | 0 of 435 pairs qualify |
| 45 | tradeable | 23 d | 28 Oct → 20 Nov |
| 60 | tradeable | 23 d | 28 Oct → 20 Nov |
| 90 | tradeable | 23 d | 28 Oct → 20 Nov |

**Move the slider, the answer stays put.** That is the difference between a finding
and an artifact. And at a 75% confidence bar, zero windows qualify at any slider
setting — which is the honest answer for a liquid large cap, and the old version
could never say it.

`test_window_scan.py` locks all of this in: the recommendation must not track the
slider, sliders 60 and 90 must agree, net must always sit below gross by exactly
the round trip, raising costs or the probability bar can only *reduce* the number
of tradeable windows, the hold's uncertainty must be smaller than the level's
(2.63% vs 8.96% — they are not the same number and only the first matters once you
are in), and a dead-flat forecast must return "no trade" rather than invent one.

> No window in the next 90 business days is worth trading. Out of 4,005 buy/sell
> pairs, none clears a 0.30% round trip with at least a 55% chance of profit.
> Holding the whole window would net −0.25% over 125 days (−0.7% a year) with only
> a 49% chance of being up — that is a coin flip, not a plan.

A scan that always finds a trade is not a scan.

## Too many lines on the outlook chart (v26)

v24 drew 60 individual simulated paths to prove the model expects movement even
when the centre line is flat. It made the point and made the chart unreadable.

Same information, three shapes instead of sixty lines: nested ribbons holding 50%,
80% and 95% of the simulated outcomes around the median — the graded fan central
banks use for exactly this problem. The spread, its growth and its asymmetry are
all still visible; the spaghetti is gone.

## Bug sweep: the IPO Center, the metals, and a gate I got wrong (v25)

Three reported faults. All three were real, and two of them were mine.

### 1. Copper showed nothing at all

    HG=F [1d, current]: 1 candle(s) where High is below Open/Close
    -> st.stop() -> blank page

The Phase 9 validation gate I added in v23 was right in principle and wrong in
calibration. "Refuse rather than substitute" is correct when the answer would be
wrong; it is the wrong response to **one malformed candle in 3,203** from a thin
futures feed. A page that refuses to load teaches you to distrust a tool that was
actually fine.

The same gate was also blocking:

| symbol | why it was refused | verdict |
|---|---|---|
| HG=F copper | 1 bad candle | repair it |
| EURUSD=X | 97 bad candles of 3,316 | repair it |
| **CL=F WTI crude** | **"non-positive prices present"** | **that is real history** |

The third is the embarrassing one. WTI settled at **−$37.63 on 20 April 2020**.
My check called the most famous print in oil-market history invalid data.

Now there is a repair step before validation. It clamps `High = max(O,H,L,C)` and
`Low = min(O,H,L,C)` — a candle's high is only ever raised to a price that candle
already contained, so nothing is invented — reports what it touched, and leaves
negative futures prices alone. Genuinely fatal problems (no data, missing columns,
unsorted timestamps, a candle dated after the as-of moment) still stop the page.

### 2. The IPO Center was a traceback

    KeyError: "['updated'] not in index"      ipo_page.py:242

IPOWatch stopped publishing the "GMP updated" column. The page selected nine
columns by name, pandas raised on the missing one, and the entire IPO Center died
— including the parts that had nothing to do with that column.

That is a bug class, not a bug: anything built on a scraped third-party table will
meet a changed schema again. Both hard-coded selections now go through a helper
that keeps the columns that exist and reports the rest:

> Not shown: updated — the source stopped publishing it. Everything else is live.

All three tabs and the full IPO analysis (verdict, scorecard, strengths, risks)
verified working in the browser.

### 3. Every symbol the UI offers, actually loaded

A catalog entry is a promise: the button is there, so clicking it must produce an
analysis. `test_catalog.py` walks all 129 symbols and checks that promise end to
end — load, repair, validate, profile.

    ok 129   repaired/warned 26   thin 0   dead-or-blocked 0   of 129

It started at **5 broken**:

* **XAUUSD=X, XAGUSD=X** (gold/silver spot) — delisted by the provider, no data at
  all. Removed from the catalog; GC=F and SI=F cover the same markets. Offering a
  button that cannot work is its own bug.
* **KC=F (16%), CT=F (12%), JPYINR=X (14%)** — refused by my repair threshold.
  Checking rather than assuming: their **Close series are clean** — no NaNs, no
  absurd returns, correct last price — and the forecast is built on Close. Only
  the intraday high/low is unreliable, which is what a thin futures or cross feed
  looks like, not a broken one. Refusing them denied three real markets over a
  defect in a few indicators.

So the threshold moved to 25%, and a high repair rate is now a **targeted** warning
instead of a refusal:

> 16% of candles needed repair — this feed's intraday high/low is unreliable, so
> ATR, candlestick patterns and range-based volatility should be treated with
> caution. The closing prices are clean, and the forecast is built on those.

Above 25% the data is still not repaired, and validation refuses it — verified
against a synthetic 30%-malformed feed.

### What "graphs are not loading" actually was

Partly the copper gate above, partly patience. Measured on an idle machine, first
analysis of a new symbol takes **70–98s** (MSFT 98s, copper 70s) for five horizons
of greedy feature-group selection, then it is cached. The same fit measured 454s
while the browser and server were competing for the same cores — which is what it
feels like in practice.

I tried to fix it and failed honestly. The five horizons are independent, so they
should parallelise:

| | baseline | threads | threads, inner n_jobs=1 | inner n_jobs=2 |
|---|---|---|---|---|
| MSFT | 97.8s | 87.4s | 91.1s | **58.3s** |
| HG=F | 70.0s | 73.0s | 103.5s | 75.4s |

A 1.68x win on one symbol and a 0.93x loss on the other is not a speedup, and
constraining the thread budget changed the forecasts (up to 0.04pp) because it
reorders floating-point reductions. **Rejected** — see `exp_speed.py`. The page
already says what it is doing while you wait.

### Verified in the browser, not just asserted

MSFT, Gold (GC=F), Silver (SI=F) and Copper (HG=F) each render fully — 10+ charts
including the 60-path forecast fan — and the IPO Center renders all three tabs
plus a full analysis. Audit: 7/7 runtime tests pass, 0 critical. Timeframe suite:
all checks pass. 46 modules compile.

## Why the forecast line is flat, answered with numbers (v24)

The complaint, again: the 21-day outlook for MSFT is a dead-flat dashed line.

### What the model actually says

    val_rmse        0.075963
    val_naive_rmse  0.076106     it beats "unchanged" by 0.19%
    alpha           0.097        so 90% of its view is shrunk away
    groups_used     ['base']     every optional feature group was rejected

`predict()` returns `alpha * blended_signal`. Alpha is fitted so the shrunk signal
beats a no-change forecast and is forced to zero unless it wins in 2 of 3 folds.
For a liquid large cap at one month it barely wins, so the centre lands near
today's price. That is the model reporting the truth about monthly large-cap
returns, not a bug.

### Two attempts to make it move, both rejected

Shrinking toward **zero** assumes the no-skill forecast is "the price does not
move". It isn't — a stock's unconditional 21-day return is positive. So the
principled fix is to shrink toward the training-period drift instead of the
origin: `y = mu + a * (f(x) - mean(f))`, with mu estimated inside each fold and
never seeing the evaluation block. 16 symbols x 3 horizons, held-out tail, purged.

| skill vs naive (pts) | zero | drift | drift + James-Stein | drift only |
|---|---|---|---|---|
| mean | **+0.69** | +0.03 | +0.12 | −0.19 |
| median | 0.00 | **+0.42** | +0.30 | +0.29 |
| worst case | **−4.7** | −25.5 | −10.7 | −25.5 |

Drift helps the *typical* symbol and occasionally fails catastrophically —
ITC.NS lost 25 points at three months because its training-period drift simply
did not persist. Win rate against the current behaviour: 25 of 48, a coin flip.

Round two gated the drift on whether it was real: `t = mu / (sd / sqrt(n_eff))`
with `n_eff = n/h`, because overlapping h-day returns on daily rows otherwise
overstate significance by sqrt(h).

| | zero | drift | gate t>2 | gate t>1 | gate t>1 + stable sign |
|---|---|---|---|---|---|
| mean skill | **+0.69** | +0.03 | +0.65 | −0.38 | +0.60 |
| worst | **−4.7** | −25.5 | −4.7 | −25.5 | −10.7 |
| beats current | — | 25/48 | 2/48 | 11/48 | 6/48 |

No variant beats what is already there. The t-gate does not separate the drifts
that persist from the ones that don't — `gate t>1` still carries the −25 point
failure. **Rejected, twice.** The flat centre stays because the evidence says so.
The scripts are in `exp_drift_target.py` and `exp_drift_gated.py`; rerun them.

### What was actually wrong: the drawing

A dead-flat line reads as "nothing will happen". The model is saying the
opposite. Its own numbers for MSFT over 21 days:

    sigma        7.6%
    q_hi_ratio   +1.59 sigma      the band is asymmetric --
    q_lo_ratio   -1.07 sigma      the upside tail is 48% longer

and the block-bootstrap built from that:

    median ABSOLUTE 21-day move        4.67%
    paths ending >3% from today        65%
    day-21 spread       p10 -8.31%   median +0.38%   p90 +9.61%

None of which the eye could see. The outlook chart now draws 60 individual
simulated futures behind the cone. Every one of them moves, they disagree about
direction, and their middle is flat — which is the honest message: **expect
movement, not a callable direction.** The chart says so in a caption rather than
leaving a flat line to be misread.

## Point-in-time architecture and the historical-price bug (v23)

### The bug, reproduced before it was fixed

Setting a past *History end date* made every price on the page become today's
price. The cause was one line: the live quote was overlaid onto the price frame
so the header could tick, and that overlay appended a row for **today** whenever
the frame ended earlier.

    frame ending 2026-08-14  ->  last candle 2026-08-14, close 727.00
    after the live overlay   ->  last candle 2026-09-28, close 722.40

Everything downstream inherited it: the KPIs, the chart, the forecast's
reference price and the scenario range were all anchored on today's market
while claiming to describe a date six weeks earlier.

### The fix: three kinds of data, separated at the architecture level

`market_data.py` now defines them explicitly:

| | what it is | mutable? |
|---|---|---|
| `CURRENT_DATA` | the market right now | yes, refreshed every few seconds |
| `HISTORICAL_DATA` | the market up to a chosen moment | **immutable** |
| `FORECAST_DATA` | what a model expects next | never mixed into either |

`get_market_data(ticker, start, end, interval, as_of)` truncates at the as-of
moment, tags the frame (`mode`, `as_of`, `immutable`) and validates it.
`apply_live_quote_safe()` refuses to touch a frame marked historical and
*reports* the refusal instead of silently proceeding.

In the dashboard a past end date now switches the whole app into **historical
mode**: the live quote is off, the auto-refreshing header and intraday panel are
disabled, and the two prices are shown side by side so they can never be
confused again (Phase 76):

    HISTORICAL MODE — as of 14 Aug 2026
    Close on 14 Aug 2026   $495.40    the selected historical date
    Price right now        $516.17    live market, shown for reference only
    Moved since then       +4.19%     not used by anything below

### Validation layer (Phase 9)

`validate_market_data()` runs before data reaches any model and checks: rows
present, columns present, timestamps sorted, duplicates, **no candle dated after
the as-of moment**, High ≥ Open/Close, Low ≤ Open/Close, positive prices,
non-negative volume, timezone consistency. On failure it raises and the
dashboard stops with an explanation — it never substitutes another date, ticker
or timeframe.

### Three more contaminated paths, found by looking rather than assuming

Fixing the loader was not the end of it. Running the page in historical mode and
reading every number on it turned up three more places where today leaked in:

**The hero header.** It runs in its own auto-refresh fragment and re-applied the
live quote itself, using the old unguarded overlay. So the banner correctly said
"as of 14 Aug 2026" while the headline underneath still read $516.17 — today's
price — with today's day range and 52-week position. The header now goes through
the same guard, and reads $495.40 with that date's range. The unguarded overlay
has been deleted rather than left in the file, because leaving it is how the bug
comes back.

**The research report.** `gather_company_data` fetched `period="max"`, i.e.
always up to today, so every price-derived judgement — valuation against its own
history, momentum, distance from the 52-week range — described today's market.
It now takes an as-of moment: the same report that valued HDFCBANK at 722.85
(today) values it at 727.00 when replaying 14 Aug 2026. Statements cannot be
rewound — Yahoo publishes no vintages — so the report says they are latest-known
instead of pretending they are point-in-time.

**The macro basket.** `_close_frame` asked the provider for data up to the end
date, which is advisory. It is now truncated on arrival, so an overhanging row
cannot put tomorrow's dollar index into today's features.

One thing I suspected and was wrong about: MSFT's 1-month return showed +23.51%
in historical mode, which looked like contamination. It is real — MSFT gapped
from 390 to 451 on 30 July 2026 earnings. Checked against the data rather than
assumed.

### The audit (Phases 1 and 83)

`audit.py` is an audit that runs rather than describes. Static: who fetches price
bars outside the engine, which cache keys cannot separate two requests, where the
live quote touches a frame, where timezones are handled locally. Runtime: seven
tests against the live code.

    TEST 1 historical end date returns that date's price          PASS
    TEST 2 two different historical dates differ                  PASS
    TEST 3 live quote refused on a historical frame               PASS
    TEST 4 no cross-ticker contamination                          PASS
    TEST 5 timeframes return genuinely different data             PASS
    TEST 6 no feature/target leakage (max |r| = 0.33)             PASS
    TEST 7 macro/market data respects the as-of date              PASS

    17 findings, 0 critical, 0 high  -> exit 0

It exits non-zero on any critical finding, so it can gate a release.

The eight HIGH findings it originally raised are closed. They were cache keys
that could not distinguish a 15-minute request from a daily one. `interval` is
now part of the key **and** changes what is loaded, so the two can never be
confused; the two functions where a timeframe is genuinely meaningless (a live
quote is one instant, fundamentals are not bars) carry a written exemption at the
definition, and the audit still reports them so the exemption stays visible
rather than becoming a silent hole.

What remains is three MEDIUM notes — the macro basket, the IPO page and the
research report still fetch bars directly rather than through the engine. All
three now respect the as-of moment, so this is an architectural tidy-up, not a
correctness bug. The audit separates bar fetches from metadata fetches
(`.info`, `.news`, earnings dates), because metadata has no timeframe and routing
it through a price engine would buy nothing.

### Honest status against the 94-phase specification

Done and verified: 1 (audit), 2 (the historical bug, in all four places it
lived), 3 (partial — the engine exists, three modules still fetch bars directly
though all respect as-of), 4-8 (timeframes, aggregation, sessions, cache keys),
9 (validation), 11 (return-based targets), 12 (purged walk-forward), 16
(out-of-fold stacking), 44-45 (feature engine and gated selection), 47 (HAR-RV
volatility), 48-49 (distributional forecast, conformal-style bands with measured
coverage), 55 (analog engine), 56 (pooled model, measured and gated off), 66-67
(feature and model ablation), 76 (current vs historical never presented as the
same number), 87 tests 1-5 and 8.

Not yet done: 13 (nested tuning), 19-20 (formal regime engine), 23 (point-in-time
macro vintages — the basket respects the as-of date but uses revised values, not
the values as first published), 24-41 (the full news intelligence system —
currently sentiment, volume and social only, not entity linking, event
classification, novelty, surprise or news replay), 42 (point-in-time
fundamentals, which the provider does not publish), 50-54 (probability
calibration, meta-labelling, abstention, drift monitoring), 60 (champion/
challenger), 61-65 (walk-forward trading simulation and automated replay at
scale), 71-72 (data and forecast quality scores), 78-82 (monitoring dashboards
and experiment tracking).

That list is deliberately explicit. The parts that are done were measured; the
parts that are not are not claimed.

## Candle timeframe selector (v22)

Two separate controls in the sidebar, because they answer different questions:

    Candle / timeframe   1m · 5m · 15m · 30m · 1h · 2h · 4h · 1d · 1wk
    Forecast horizon     depends on the candle (next candle … 1 year)

Picking a candle size changes the **analysis**, not a label. `test_timeframes.py`
proves it on every run:

| check | result |
|---|---|
| candle spacing matches the interval | PASS for 5m / 15m / 1h / 1d |
| candle counts differ | 4,200 / 1,425 / 5,027 / 3,147 |
| 15m candles are the true aggregate of their 5m candles | 38/38 matched on high and low |
| RSI(14) recalculated per timeframe | 17.3 / 28.4 / 36.4 / 49.8 |
| ATR recalculated per timeframe | 0.22% / 0.35% / 0.61% / 1.78% |
| support & resistance differ | PASS |
| forecast timestamps step by the candle size | 15m → 00:15, 1d → 1 day |
| forecasts differ | −0.19% (15m) vs +0.26% (1d) |
| model cache keys differ | `HDFCBANK.NS_15m_1425` vs `HDFCBANK.NS_1d_3147` |

### What the provider can actually serve

Every limit was measured, not assumed (HDFCBANK.NS, Sep 2026):

| interval | real history | bars | how it is obtained |
|---|---|---|---|
| 1m | 7 days | ~2,500 | native |
| 5m / 15m / 30m | 60 days | 4,274 / 1,450 / 756 | native |
| 1h | 730 days | 5,034 | native |
| 2h / 4h | 730 days | 2,874 / 1,440 | **resampled from 1h, within each session** |
| 1d / 1wk | full history | 7,720 / 1,604 | native |

Yahoo's native 4-hour feed only reaches back 60 days — 117 bars, far too few to
model — so 2h and 4h are built by resampling hourly bars inside each trading
session (a candle never straddles the overnight gap). The UI labels them
"resampled" and the page says so. Nothing else is substituted: a timeframe the
provider cannot serve with enough history is not offered at all.

### Horizon is measured in candles, and capped honestly

"21 trading days" is 21 candles on a daily chart and 525 on a 15-minute one.
The page shows both, and refuses horizons the data cannot support:

> **1 month is 525 candles at 15 minutes, and only 1,425 candles of history
> exist.** A forecast that far out would rest on roughly 3 independent
> examples, so it is capped at 71 candles. For a longer view, switch to a
> larger candle.

Sessions are exchange-specific: an NSE session is 375 minutes (25 fifteen-minute
candles), a US one 390 (26), and crypto runs 24/7 — the arithmetic follows the
symbol.

### What is deliberately excluded intraday

Daily news tone, macro series, earnings dates, the daily social reading, the
universe model, the fundamental research report and the HAR volatility model are
all one observation per day. Spreading a single value across 25 intraday candles
would present one fact as 25 pieces of evidence, so on an intraday timeframe
those groups are dropped and the page states it. Models, feature selection and
caches are keyed by timeframe, so a 15-minute model is never served to a daily
request.

## Fitting the combination instead of averaging it (v22)

The complaint that started this: a one-month forecast identical to today's price.
Tracing IDEA.NS showed exactly why — the three ensemble members disagreed
violently (Ridge −4.31%, Random Forest +0.79%, XGBoost +1.21%), the
inverse-RMSE weights came out as a plain average (0.33/0.34/0.33), the average
was noise, and the validation gate correctly zeroed it.

Two changes address it:

1. **Stacking** — non-negative least squares on the members' out-of-fold
   predictions, so the engine can say "trust XGBoost, ignore Ridge" instead of
   averaging models that contradict each other.
2. **The indicator bank as six gated groups** (trend, momentum, volatility,
   volume, structure, seasonality) offered to the price model.

Both were measured, and the first attempt **failed**: −0.48 skill points and
−0.65 direction points across 26 held-out cases, with TCS.NS losing 10 points
even though ITC gained 4. The gates were letting through more harm than good.

Tightening the bar for these specific additions — every fold must improve, and
by 1% rather than 0.3% — flipped it:

| | loose gate | strict gate |
|---|---|---|
| skill change | −0.48 pts | **+0.10 pts** |
| direction change | −0.65 pts | +0.05 pts |
| stacking chosen | 18 of 26 | 3 of 26 |

Net positive, and now the movement is earned rather than manufactured. On full
history IDEA.NS's one-month forecast goes from a flat 0.00% to **+1.29%**, with
the stacked combination and two indicator groups accepted; MSFT stays at +0.39%
because nothing in its data justifies more. The summary table in Model Insights
shows which combination each horizon chose.

## Attacking what is actually predictable (v21)

Returns are close to unforecastable at a one-month horizon; that was established
in v20 and no amount of extra models changed it. So this round went after the
two things that *are* learnable -- **volatility** and **the shape of the
uncertainty** -- and tested one structural idea that could have changed the
picture entirely: **learning across a whole universe instead of one symbol**.

### Shipped: asymmetric, volatility-adaptive prediction bands

Five band constructions were compared on the same fits (36 held-out
symbol/horizon cases), so only the uncertainty layer differed:

| band | mean distance from its promised 80% coverage |
|---|---|
| fixed width per horizon (original) | 9.58 pts |
| ±z·σ scaled by trailing volatility | 4.64 pts |
| ±z·σ scaled by the HAR volatility forecast | 6.02 pts |
| **empirical 10th/90th percentiles × trailing volatility** | **3.64 pts** |
| empirical percentiles × HAR forecast | 5.18 pts |

The winner does two things a fixed Gaussian band cannot: it **breathes with the
market** (tighter in calm tape, wider in turbulent), and it is **asymmetric** --
the edges come from the 10th and 90th percentiles of the model's own past
misses, so a symbol that historically fell harder than it rose gets a longer
tail downwards. MSFT's current one-month band is −6.3% / +10.6%, not ±8.5%.

### Shipped: a volatility model, reported rather than hidden

`volatility_model.py` implements **HAR-RV** (Corsi 2009) -- realised volatility
over daily, weekly, monthly and quarterly windows, plus leverage terms for the
way volatility jumps after falls, fitted in log space so it cannot go negative.

Against the "next month looks like last month" baseline, on validation data:
**+12.1% average error reduction, better in 22 of 24 symbol/horizon cases**
(+22.5% for MSFT at one month). It refuses to fit at all where it cannot beat
that baseline, which is why it declines on WTI crude at 21 days.

Note the honest split: HAR **forecasts volatility better** but **calibrates
bands worse** (6.02 vs 4.64 pts) -- its smooth estimate misses the spikes that
coverage depends on. So it is reported as a volatility outlook in the Scenario
Lab ("next month 24.2%, rising vs 22.0% now") and kept out of the band path.
Two different jobs, two different measurements, two different winners.

### Shipped: real risk numbers

From the simulated paths: **Value at Risk (95%)**, **expected shortfall** (the
average outcome inside that worst 5%), the **probability of touching −10% or
−20%** at any point in the window, and an **upside/downside ratio**. These are
the numbers a risk desk actually looks at, and they come from the same
distribution the fan chart draws.

### Built, measured, and NOT shipped: the universe model

`pooled_model.py` trains one model across **38 symbols / 113,041 rows**, on the
theory that a per-symbol model starves on ~2,500 rows (only ~120 independent
outcomes at a 21-day horizon). Every feature in this project is stationary by
construction, so a row from RELIANCE and a row from MSFT are directly
comparable. It also adds cross-sectional context -- each feature's percentile
rank within the universe that day -- which a single-symbol model can never see.

The first test looked like a breakthrough:

| | per-symbol engine | pooled model |
|---|---|---|
| skill vs naive, 21 days | −1.14% | **+1.58%** |
| direction accuracy | 50.6% | **55.1%** |
| rank-IC | — | **+0.115** |
| shrinkage α | ~0.10 | **0.55** |

An α of 0.55 would have produced visibly non-flat forecasts. Before shipping it,
its **out-of-fold history** (330,400 predictions, cross-fitted through time so no
row is scored by a model that saw it) was checked year by year:

| year | rank-IC, 21 days | direction |
|---|---|---|
| 2019 | −0.129 | 50.3% |
| 2020 | −0.033 | 48.7% |
| 2021 | +0.078 | 56.1% |
| 2022 | +0.049 | 49.8% |
| 2023 | **+0.223** | 62.0% |
| 2024 | −0.014 | 52.3% |
| 2025 | +0.071 | 56.2% |
| 2026 | +0.113 | 55.4% |

All-years rank-IC: **+0.038**, unshrunk skill **−1.70%**, positive in **5 of 8
years** with 2023 doing most of the work, and **no edge at 5 days** (IC +0.008).
The +1.58% came from one favourable split.

Independently, the engine's own walk-forward gate had already rejected it: when
the pooled view was offered as a feature group, it was kept in **0 of 22** fits,
and for MSFT its correlation with actual outcomes over the full out-of-fold
history was **negative** (−0.056).

So it is **off by default**, behind a sidebar switch labelled with those numbers.
The plumbing stays because it is self-gating -- the engine drops it per symbol
when it does not help -- and because the measurement itself is the useful
artifact. It also does not transfer at all outside its training universe
(−1.11% skill on gold, crude, bitcoin, FX), which the coverage filter handles
automatically.

### The pattern across v19-v21

Four structural ideas were tested with the same protocol. Three were rejected:
drift in the point forecast (−0.29 skill), indicators in the price regression
(one −351 blow-up), and the pooled universe model (unstable across years). Two
were adopted on evidence: the indicator bank in the *direction* model
(53.9% → 55.1%, never worse) and the calibrated adaptive band (9.58 → 3.64 pts).
One was adopted for reporting only: HAR volatility.

That ratio -- three rejections for every two adoptions -- is what a real research
process looks like. A system that adopts everything it tries is not measuring.

## The accuracy programme (v20): a measurement bench, 71 indicators, 8 learners

The brief was "make the prediction as good as it can possibly be, and stop the
one-month forecast looking like today's price". That was treated as a research
question rather than a coding task, because the only way to know whether an idea
helps is to measure it on data the fitting never saw.

### The bench: `model_lab.py` + `lab_configs.py`

Every idea is judged identically:

    |<------------- fit / select ------------->|  purge h days  |<-- test 15% -->|

The last 15% of history is untouched while fitting, selecting features and
tuning; rows whose h-day outcome overlaps the test window are purged so no label
leaks across the boundary. Four numbers come out: **skill** (RMSE vs the naive
"price stays put"), **direction accuracy**, **cover80** (does the 80% band
actually contain 80% of outcomes?) and **move** (is the model saying anything at
all?). The panel is 12 symbols across US and Indian equities, an index, gold,
crude, bitcoin, an ETF and a currency pair — an idea that only helps US mega-caps
is not an improvement.

**The bench caught its own bug first.** The first run reported 99.7% skill and
100% direction accuracy at h=1 — impossible. The feature table carries legacy
helper columns (`Target_Return` has correlation 1.000 with the answer) that the
engine never sees because it works off its registered feature groups; the lab was
reading all 93 columns instead of the engine's 53. Fixed before any conclusion
was drawn. This is exactly why the bench exists.

### What was built and tested

| Candidate | What it is |
|---|---|
| `features_indicators.py` | **71 indicators** written from source formulas, all stationary by construction: ADX/DMI, Aroon, Ichimoku, ATR channels, linear-regression slope and R², RSI (Wilder), Stochastics, Williams %R, CCI, TSI, KAMA efficiency, MACD/PPO, Bollinger %B and width, Keltner, Donchian, Ulcer, Parkinson and Garman-Klass volatility, OBV, CMF, MFI, Amihud illiquidity, VWAP distance, Hurst exponent, sample entropy, 52-week position, drawdown, skew/kurtosis, and seasonality (day-of-week, month, turn-of-month, quarter end) |
| `+MODELS` | a wider bench: HistGradientBoosting (scikit-learn's LightGBM-equivalent), ExtraTrees, ElasticNet, Huber, kNN alongside Ridge/RandomForest/XGBoost |
| `+VOL` | predict *return ÷ trailing volatility*, then rescale — a homoscedastic target |
| `+WEIGHTS` | exponentially decayed sample weights (2-year half-life) |
| `+CONFORMAL` | empirical residual quantiles instead of a Gaussian band |
| `+DIRECTION` | a classifier's P(up) mapped to an expected return via historical up/down sizes |
| combinations | VOL+IND, VOL+IND+W, VOL+IND+W+M, VOL+IND+W+C |

### What the measurements said (36 held-out symbol/horizon cases)

| config | skill | direction | 80% band coverage |
|---|---|---|---|
| control | −1.14 | 50.6% | 72.9% |
| +IND | −10.19* | **52.9%** | 74.4% |
| +VOL | −0.78 | 51.1% | **81.0%** |
| +MODELS | +0.10 | 51.0% | 74.6% |
| +WEIGHTS | +0.25 | 50.7% | 74.7% |
| +DIRECTION | −2.68 | 51.8% | 72.7% |
| VOL+IND | −0.64 | **52.6%** | **79.8%** |

\* the −10 average is one blow-up (GOLDBEES.NS at one month, −351); every other
case sits between −13 and +5.7, and +IND was actually better in 18 of 36.

Nothing improved squared error. Two things clearly improved something else, and
those two shipped.

### Shipped 1 — volatility-adaptive prediction bands

The old band was one width per horizon, averaged over all past conditions: too
narrow in turbulent markets, too wide in calm ones. A band advertised as 80% was
containing only **72.9%** of outcomes (71.4% at one month) — quiet overconfidence.

The engine now measures each miss *in units of the volatility that was present
that day* (`sigma_ratio = std(residual / trailing_vol)`) and rebuilds the band
from **today's** volatility, clamped to 0.3–3× the static width so one freak
reading cannot distort it.

Measured A/B on the same fits, so only the interval changes:

| horizon | mean miss from 80%, fixed | adaptive |
|---|---|---|
| 1 day | 7.33 pts | **3.44** |
| 5 days | 9.20 pts | **2.82** |
| 21 days | 12.01 pts | **5.82** |
| **overall** | **9.51 pts** | **4.03 pts**, closer in **29 of 36** cases |

A visible side effect the brief asked for: the forecast is no longer identical
looking everywhere. The band now breathes — in a calm tape it tightens, in a
turbulent one it widens — and the dashboard says which is happening and why.

### Shipped 2 — the indicator bank inside the direction model

Indicators carry information about *which way* the market goes, not *how far*, so
they were given to the classifier rather than the price regression, and kept only
where they cut validation log-loss (the same gate news goes through).

| | old | new |
|---|---|---|
| mean direction accuracy | 53.86% | **55.06%** |
| accuracy on confident days | 55.17% | **56.10%** |
| beats "always guess up" | 7/12 symbols | **8/12** |

Better in 5 of 12 symbols (NIFTY +4.1, ITC +3.9, MSFT +2.8, EURUSD +2.4, WTI
+1.3) and **never worse** — where indicators do not help, the gate rejects them
and the model is unchanged.

### Not shipped, and why

* **Drift in the point forecast** (v19): flattered validation, lost 0.29 skill
  points on held-out data. Rejected.
* **Indicators in the price regression**: better direction, worse squared error,
  one catastrophic blow-up. Rejected for that job, used for the other one.
* **Wider learner bench / recency weights**: +0.10 and +0.25 points, inside the
  noise, at real cost in fit time. Not adopted; the code stays in `lab_configs.py`
  so the result is reproducible.
* **Plain conformal bands**: identical skill, slightly worse coverage than the
  volatility-scaled version that shipped.

The honest summary: **no feature set and no model family made a liquid market's
one-month return predictable.** What can be improved — and was — is the honesty
and usefulness of the uncertainty around it, and the direction call.

## Scenario lab (v20)

Because the central forecast is close to flat by design, `scenarios.py` shows the
distribution instead of the average:

* **3,000 simulated paths** built by block-bootstrapping the symbol's own daily
  returns in 10-day blocks, so real volatility clustering and fat tails survive
  (a plain normal random walk badly understates crashes). The paths are centred
  on the engine's forecast and scaled to its validated uncertainty, so the fan
  agrees with the 80% band rather than offering a second opinion.
* **Odds that matter**: chance of finishing higher, of ±5%, the typical peak and
  dip along the way, and for any target price you type — the chance of touching
  it and the median number of days to get there.
* **Historical analogs**: the 40 past days whose trend, volatility, momentum and
  drawdown most resemble today, and what actually happened next. For MSFT that is
  72% positive with a +1.1% median; for bitcoin, 38% positive with a −3.4%
  median. Descriptive, not predictive — but if the simulation and the analogs
  disagree, that disagreement is information.

Every path in that fan moves. It is the *average* of them that lands near today,
which is precisely why a single number looked flat.

## The 10-second version (v20)

A summary card at the top of the page states, in plain English, what the whole
dashboard concluded: today's move, the central forecast and why it is small, the
simulated odds and good/bad cases, the technical reading, the research verdict
and the current news and social mood — each claim expanded in a section below.

## Why the 1-month forecast looks like today's price (v19)

A fair question: every symbol's 1-month forecast sits within a fraction of a
percent of the current price. That is the shrinkage doing its job, not a broken
pipeline - but it was worth testing properly.

**What the engine does.** Each horizon's prediction is `alpha x ensemble view`,
where `alpha` is fitted on three walk-forward validation folds and forced to 0
unless the model beat the naive "no change" forecast in at least 2 of them. For
MSFT the raw ensemble wanted +4.9% over 21 days; validation said its skill was
0.18%, so alpha came out 0.10 and the shown forecast was +0.47%. For WTI crude
alpha was 0 on every horizon - three folds, no win, so the model defers entirely.

**The fix that was tried and rejected.** Shrinking toward zero implicitly says
"this asset will not move", when the neutral assumption should arguably be that
it keeps drifting as it has (a random walk *with* drift). That was implemented -
`beta x drift + alpha x (view - drift)`, drift measured on training data only,
capped at +/-30% a year, with beta validated on the same folds. On the validation
folds it looked excellent: MSFT's 21-day skill went 0.18% -> 1.76%, gold's
1.11% -> 2.87%, and forecasts finally sloped (+1.2% for MSFT, +1.0% for gold).

On the **held-out test set** it was worse: mean -0.29 skill points across 24
symbol/horizon cases, better in 7, worse in 14 (TCS 21-day -8.95% -> -13.73%).
Shrinking the drift by its own statistical significance (James-Stein) did not
rescue it: mean -0.35 points, better in 5 of 24. The reason is simple - a decade
of average return says little about the next month, and the recent test window
was mostly a downtrend for the Indian names, where adding upward drift hurt most.

So the change was **reverted**. The validation-fold numbers were flattering
because the same folds fitted the coefficients; the held-out test is the honest
one, and it said no.

**What was added instead.** The forecast ladder now shows the two things that do
carry information at these horizons, both from the model's own fitted
uncertainty: the **chance of finishing higher than today** and the **typical
swing**. MSFT reads "1 month: $502.96, +0.47%, 52% chance higher than today,
typical swing +/-9.8%". A 52% chance with a +/-10% swing is a genuinely different
proposition from 52% with +/-2%, and neither is visible in a point estimate.

## Too-new stocks still show the live market (v18)

A symbol with less than 300 trading sessions cannot be forecast (the indicators
alone need a 200-day average, and each horizon's model needs 300 rows with a known
outcome), so the dashboard says so instead of failing. It now also renders the
**live intraday panel underneath that message** - price, day range, volume, the
minute chart and the auto-refresh - because a stock that listed this morning is
trading right now even though nothing can be trained on it yet.

Implementation note: `_live_body()` and the weekend-handling flags moved above the
"freshly listed" branch in `dashboard.py` so both paths can render the same panel.

## Every score explains itself (v18)

The 0-10 rings were a black box: you saw "Valuation 1.6/10" with no way to ask why.
Each ring is now a click-to-expand panel (a native `<details>` element, so it opens
instantly with no Streamlit rerun) listing every metric that fed it, the value, and
that metric's own score:

* **Stock analysis -> Research Report** - e.g. clicking *Valuation* on MSFT shows
  P/E 27.9 (4.0/10), forward P/E 21.0 (6.3/10), PEG 1.6, EV/EBITDA 19.3, price/book
  8.4 (1.9/10), FCF yield 1.8%, P/E vs its own history 86% (8.8/10), and names the
  metric holding the score back.
* **Commodities / FX / crypto / ETFs** - the same for Trend, Momentum, Volatility,
  Drawdown, Relative strength and Consistency.
* **IPO Center** - every pillar, plus the two headline rings: *Listing-gain view*
  explains that it is built from GMP and final subscription (QIB weighted highest),
  *Long-term view* that it averages Growth, Profitability, Valuation and Issue quality.
* **Sidebar quick picks** - an "What are these?" panel explains each category
  (what a futures contract, a currency pair or a tracking ETF actually is) and what
  every symbol in it stands for.

Implementation lives in `ui_theme.rings(pillars, why=...)` and `ui_theme.why_html()`;
the dashboard builds the breakdown from the same `sub_scores` / `metrics` the
scorecard tables use, so the explanation can never drift from the score.

## Social media buzz + live headlines (v17)

`social_media.py` reads what people are posting about the stock **right now** and
shows it in section 04 of the dashboard, next to the headlines that can move it.

| Platform | How it is read | Works without a key? |
|---|---|---|
| **StockTwits** | Public API, ~30 latest messages for the symbol. Each message can carry the author's own **Bullish / Bearish** tag, which beats guessing sentiment from text | Yes (US-listed symbols; Indian tickers have no board) |
| **Reddit** | Public search **RSS** feed (`/search.rss`) — the `.json` endpoints return 403 to anonymous clients. Rate-limited, so calls are spaced 4 s apart and retried once | Yes |
| **YouTube** | The `ytInitialData` blob on the search page, sorted by upload date: title, channel, age, view count | Yes |
| **X / Twitter** | `api.x.com/2/tweets/search/recent` — X switched to pay-per-use in Feb 2026 and has **no free tier**: reads cost about $0.005 per post, billed from credits at console.x.com. Paste a Bearer token in the sidebar (or set `X_BEARER_TOKEN`) to include X. To keep the bill tiny the dashboard asks for 10 posts at most **once an hour** (about $1.20 a day if left open) plus a manual "Fetch X now" button; without a token the panel just says the platform is optional | No — paid |

Everything is scored with the same VADER + finance lexicon used for news, and
Reddit/YouTube hits are filtered for relevance (the symbol as a whole word, or the
company name together with a market word) so "self-reliance" or "apple pie" don't
count as coverage of RELIANCE or AAPL.

What you see: overall social mood, posts tracked, bullish share, a card per
platform, the most recent posts (clickable, with author and age), and a
**Headlines** tab listing current news with per-headline sentiment. The panel
re-fetches **every 60 seconds on its own**, and a "Social mood" card also appears
in the AI snapshot.

### How it reaches the predictions — and the honest limit

Social platforms publish **no free historical archive**, so there is nothing to
backfill. The pipeline therefore works forwards:

1. Every day you open the dashboard, `collect_today()` saves one row per symbol in
   `news_cache/social_<TICKER>.csv` (mood, post count, bullish share, per-platform counts).
2. Once **40 days** exist, `build_social_features()` turns them into
   `Soc_Mood_1d`, `Soc_Mood_3d`, `Soc_Mood_Shift`, `Soc_Buzz`, `Soc_Bull_Ratio`
   (all shifted by one day, so no look-ahead).
3. The forecast engine then treats **"Social buzz" as one more feature group** and
   runs it through the same walk-forward test as market, technical, candles,
   macro, events and news — it is kept only for the horizons where it consistently
   cuts the error, and dropped everywhere else.

Until those 40 days accumulate the dashboard says so explicitly
("Model input: N/40 daily readings collected") instead of pretending the models
already use it. This is the same rule the rest of the project follows: a signal is
displayed when it is real, and used only when it is *measured* to help.

## Every market, not just shares (v16)

The dashboard used to assume every symbol was a company share. It now analyses
**commodities, currency pairs, crypto, ETFs and indices** with the same engine:

| Asset class | Examples you can search | What changes |
|---|---|---|
| Commodity | `GC=F` gold, `SI=F` silver, `HG=F` copper, `CL=F` WTI crude, `NG=F` natural gas | Market context uses the **US dollar index** instead of a sector ETF |
| Forex | `USDINR=X`, `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` | Context = dollar index + US 10-year yield; no volume is published, so volume features simply score as useless and get dropped |
| Crypto | `BTC-USD`, `ETH-USD`, `SOL-USD` | Trades 7 days a week, so forecast dates are **calendar days**, weekends are not hidden on charts, and the market is never "closed" |
| ETF | `NIFTYBEES.NS`, `GOLDBEES.NS`, `SILVERBEES.NS`, `SPY`, `QQQ`, `GLD` | Fund size, expense ratio and what it tracks replace market cap / P/E; a gold or silver ETF is benchmarked against the metal itself |
| Index | `^NSEI`, `^BSESN`, `^GSPC`, `^IXIC`, `^VIX` | Treated like a fund with no cost data |

How it works:

* `stock_search.asset_class()` classifies a symbol from Yahoo's `quoteType` plus
  the symbol's shape (`=F` future, `=X` pair, `^` index, `BTC-USD` crypto), with a
  built-in catalogue (`POPULAR_ASSETS`) overriding Yahoo where it is wrong — Yahoo
  reports Indian ETFs such as GOLDBEES.NS as plain "EQUITY".
* The sidebar has **quick picks** (Metals / Energy / Forex / Crypto / ETFs / Indices)
  and the search box matches names like "gold", "silver", "usd inr" or "bitcoin".
* The header KPI cards swap themselves: market cap / P/E / dividend yield for shares,
  fund size / expense ratio / tracked index for ETFs, and 1-week…1-year returns plus
  annualised volatility for metals, currencies and crypto.
* **The research report for a non-share is a different report.** Gold has no revenue,
  margin or P/E, so `research_analyst.analyze_asset()` scores six price-based pillars
  instead — Trend, Momentum, Volatility, Drawdown, Relative strength and Consistency —
  and adds a seasonality chart (average return per calendar month) plus the correlation
  with its benchmark. The verdict ring, bull/bear case and "what to watch" card are
  the same widgets as the equity report.

The forecasting models themselves are unchanged: they only ever needed OHLCV history,
which every one of these instruments has.

## IPO Center (v15)

Switch the sidebar to **🚀 IPO Center** (`ipo_page.py` UI, `ipo_analyzer.py` logic)
for Indian mainboard and SME IPOs:

- **Search any IPO**: live, upcoming or recently listed.
- **Per-IPO analysis**: price band, lot size and minimum investment,
  issue size, fresh issue vs offer-for-sale, GMP and estimated listing
  price, bidding / allotment / listing dates, promoter holding before and
  after, lead managers and registrar; live **subscription by category**
  (QIB, NII, Retail, Employees) from NSE; **company financials** (revenue,
  PAT, assets), KPIs (ROE, ROCE, margins, debt), **listed peer comparison**
  and objects of the issue.
- **Apply / avoid verdict** with two scores:
  - *Listing-gain view*: GMP plus subscription, especially institutional (QIB) demand
  - *Long-term view*: growth, profitability, valuation vs peers, issue
    quality (fresh-issue share, promoter holding, debt)
  -> Apply / Apply for listing gains only / Long-term only / Neutral / Avoid,
  with strengths and risks. Subscription isn't scored until the last
  bidding day (early numbers are naturally low), and a Rs.0 GMP before the
  issue opens isn't counted against it.
- **Live GMP board** (mainboard + SME), **recent listings** (listing-day
  gain vs return since IPO today) and **how reliable GMP is**: across
  ~300 past mainboard IPOs, GMP called the listing direction right about
  83% of the time (correlation ~0.87, typical miss about 7 points).

**Data sources**: NSE's public IPO pages (official lists, issue details,
subscription) and IPOWatch (GMP, financials, peers). GMP is an
**unofficial, unregulated** grey-market quote. Everything is cached (15-60
minutes) to keep requests light. Please respect these sites' terms of use.
Always read the offer document (RHP). This is not investment advice.

## Exponential smoothing with a fixed alpha (v14)

`model_smoothing.py` adds **Simple Exponential Smoothing (SES)** with a
**fixed smoothing factor alpha = 0.3** (constant `SMOOTHING_ALPHA`):

    level_t = 0.3 * price_t + 0.7 * level_(t-1)      forecast for t+1 = level_t

- **main.py**: "Exponential Smoothing (alpha=0.3)" is now a 9th model in
  the comparison table and backtest.
- **Forecast engine**: a new "Exp. smoothing (alpha=0.3)" feature group
  (price vs its smoothed level, 5- and 20-day slope of the level), tested
  like every other group and kept only if it helps.
- **Dashboard**: "Exp. smoothing (alpha=0.3)" chart overlay.

On its own SES is a weak next-day predictor (MSFT test period: RMSE 6.76
vs 5.05 for "tomorrow = today", i.e. 34% worse), because the smoothed
level lags the price on a trending stock -- 70% of each update's weight
sits on the past. That's a known property of fixed-alpha smoothing and a
useful point for a report; it's more useful as a trend feature.

Smoothing constants elsewhere in the project (all fixed): EMA 5/10/20/50
features use alpha = 2/(span+1) = 0.333 / 0.182 / 0.095 / 0.039; MACD uses
alpha = 0.154 (12-day), 0.074 (26-day) and 0.200 (9-day signal).

## Research analyst, live chart & animations (v13)

- **🧾 Research Report tab** (`research_analyst.py`): a rule-based equity
  research analyst. It reads the company profile and leadership, 4-5 years
  of income statements, balance sheets and cash flows, valuation ratios,
  ownership, analyst ratings, upcoming earnings/dividend dates and the full
  price history since listing, then:
  - scores 7 pillars out of 10: Growth, Profitability, Financial health,
    Valuation, Shareholder returns, Momentum, Risk (animated score rings)
  - gives an overall verdict (Strong / Good / Mixed / Weak) and a one-line
    stance (e.g. "High-quality business at a fair price")
  - writes an executive summary, strengths (bull case), risks (bear case)
    and what to watch (next earnings, ex-dividend date, weakest area, the
    price model's 1-month outlook)
  - sub-tabs: Company (profile, leadership), Financials (revenue/profit,
    margins, cash flow, equity vs debt, latest quarters, scorecard),
    Valuation (ratios, P/E vs its own history), Price history (long-run
    CAGR, calendar-year returns, drawdowns, dividends), Ownership & analysts
  - downloadable Markdown report
  Every statement maps to a metric with a transparent threshold (general
  rules of thumb, not sector-specific; financial companies skip debt
  scoring). Historical P/E uses actual traded prices; long-run returns use
  dividend-adjusted prices.
- **📡 Live market section**: intraday chart with a LIVE price taken from
  Yahoo's quote feed, which updates every ~5-15 seconds (the 1-minute bars
  behind it run about a minute behind the exchange; for NSE both are far
  fresher than the ~15 min often assumed). Choose 1-day/1-minute or
  5-day/5-minute, set the auto-refresh interval (2 / 5 / 10 / 30 / 60 s,
  default 5 s) and use "🔄 Refresh now" for an immediate update. Only this
  panel reloads, not the page.
  **Note:** Streamlit starts a fragment's auto-refresh timer only once the
  whole page has finished running, so on the FIRST load of a stock (models
  training, news downloading: 1-3 minutes) the live panel stays frozen.
  After that it ticks on its own. The manual button always works.
- **Animations**: scrolling ticker tape (pauses on hover), cards that fade
  in one after another, score rings that fill up, a slowly shifting hero
  gradient with a shimmering title, a pulsing verdict glow, a pinging LIVE
  dot, and hover lift/glow on cards. All animations switch off
  automatically if the operating system is set to "reduce motion".

## Forecast engine v12: every signal, kept only if it helps

`forecast_engine.py` now powers every price forecast in the dashboard.

**Everything is offered to the price models** (previously some of these
were display-only or used only by the direction classifier):

| Group | What's in it |
|---|---|
| Price & volume | 39 scale-free indicators |
| Market & VIX | index returns, VIX level/changes, relative strength vs market & sector, 60-day beta/correlation |
| Technical rating | the scorecard's votes computed for every past day |
| Candlesticks | pattern bias, bullish/bearish counts |
| Calendar | month, turn-of-month, days to month end |
| Macro | US 10Y yield, dollar index, oil, gold (+ USD/INR and the previous US session for Indian stocks, lagged a day so there's no look-ahead) |
| Earnings & dividends | days to/since earnings, last earnings surprise, analyst up/downgrades (US), days since dividend, trailing yield |
| News | GDELT, Alpha Vantage, own daily headlines |

**How groups are chosen**: walk-forward cross-validation over 3 past
periods; a group is kept only if it cuts error by >= 0.3% on average AND in
at least 2 of 3 periods. If a model can't beat "no change" in at least 2
periods, its signal strength is set to 0 (falls back to naive).

**Direct multi-horizon models**: separate models for 1, 5, 10, 21 and 63
trading days ahead (interpolated for dates in between) instead of stacking
daily guesses. Every forecast has an **80% likely range** from the model's
own out-of-sample errors. The Buy/Sell threshold sliders now work.

**Honest held-out results** (last 15% of days, never used for selection or
training; % lower error than "no change"):

| Stock | 1-day (v10 → v12) | 5-day | 21-day | Extra groups kept |
|---|---|---|---|---|
| MSFT | −0.84% → −0.17% | −0.17% | −0.25% | none |
| AAPL | 0.00% → +0.21% | +0.56% | −1.59% | none |
| JPM | −0.86% → −0.06% | −0.19% | **+2.98%** (vs +2.46% without extras) | technical, calendar |
| RELIANCE.NS | 0.00% → 0.00% | +0.06% (vs −0.11%) | −0.02% | technical (5-day) |
| TCS.NS | −2.10% → −0.96% | −6.79% (vs −2.07%) | −8.62% | market, candles (5-day) |
| COALINDIA.NS | 0.00% → 0.00% | 0.00% | 0.00% | none |

The 1-day forecast improved on every stock. Longer horizons are mixed: a
real gain for JPM, but TCS's sharp 2025-26 sell-off (unlike anything in
its training years) hurt. That's regime change, which no model built on
past patterns can fully avoid. An even stricter rule (must beat naive in
all 3 periods) was tested and rejected: it removed the real gains without
fixing TCS. The dashboard's Model Insights → "Held-out accuracy" button
runs this same test for whatever stock you're viewing.

**Dashboard additions**: a market ticker tape (index, VIX, sector, rates,
dollar, gold, oil), "What the AI used" chips, a forecast ladder (1 day to 3
months, each with its likely range), a 21-day outlook chart with a shaded
likely-range band, and forecast bands on the date and buy/sell charts.

## New look (v11): "StockSense" dark trading-terminal UI

The dashboard was redesigned (`ui_theme.py` + `.streamlit/config.toml`):

- **Hero header**: company name, ticker/exchange/sector chips, big live
  price with a colored change pill (▲/▼, so it's not color-only), and a
  pulsing "market open" indicator.
- **KPI cards**: day range and 52-week range as visual bars showing where
  today's price sits, plus market cap, P/E, beta, analyst target (with
  upside), volume vs. its 20-day average, 1-month and YTD return, and
  dividend yield.
- **AI Snapshot**: next-session forecast, direction model, technical
  rating, candlestick reading, news mood and model signal strength, as
  color-coded cards at a glance.
- **Charts**: one shared Plotly theme (crosshair hover, recessive grid,
  right-side price axis) with colorblind-validated series colors in fixed
  order; gains/losses use reserved green/red status colors.
- Pill-style tabs, gradient buttons, glass cards, Inter + JetBrains Mono
  fonts, a branded sidebar. Works on phones as well.
- The app name/tagline are `APP_NAME` / `APP_TAGLINE` at the top of
  `ui_theme.py`; change them to rebrand.
- **Run it from inside the `stock_project` folder** (`streamlit run
  dashboard.py`) so Streamlit picks up `.streamlit/config.toml`, which
  themes the built-in widgets (tables, inputs, sliders). Requires
  Streamlit 1.50+.

## News in the models (v10)

News is now **taken into consideration, but never relied on**:

- **Where it comes from** (`news_history.py`):
  - **GDELT** (free, no key): daily news *tone* and *article volume* for
    the company's name across worldwide online news, back to 2017. This
    is the main training source.
  - **Alpha Vantage** (optional): per-stock sentiment scores (US stocks,
    history from ~2022). Get a free key at
    https://www.alphavantage.co/support/#api-key and paste it in the
    dashboard sidebar, or set the `ALPHAVANTAGE_API_KEY` environment variable.
  - **Your own daily collection**: every time you open a stock (or run
    `collect_news.py`), today's headlines from **Yahoo Finance + Google
    News** (which aggregates Economic Times, Moneycontrol, Reuters, CNBC,
    ...) are scored and saved to `news_cache/`. After ~120 days the models
    start using this history automatically.
- **How the models use it** (`dashboard_logic.py`): news features
  (3- and 14-day tone, tone shift, unusual attention, coverage) are an
  *optional extra input* next to ~40 price features. Both the price
  ensemble and the direction classifiers are trained **with and without**
  news, and news is kept only if it lowers error on validation data. Model
  Insights → "How news is used" shows the decision and the numbers, and
  the held-out accuracy is shown both ways on the same test days.
- **No look-ahead**: a trading day's features only use news dated up to the
  previous calendar day, so time-zone differences can't leak "tomorrow's"
  news into a prediction.
- **Honest result so far**: the effect is small, e.g. MSFT held-out skill
  −0.84% → −0.59% with news, Reliance direction accuracy 53.9% → 54.5%.
  That's consistent with research: most news is priced in within hours.
- **Reliability**: GDELT's free API rate-limits hard. Downloads are cached
  and time-limited (about 90 seconds). If GDELT is busy, the dashboard simply runs
  without news and retries next time.
- **Daily collection on a schedule** (optional):
  ```bash
  python collect_news.py MSFT RELIANCE.NS
  ```
  See the top of `collect_news.py` for a one-line Windows Task Scheduler
  command to run it every evening.

## Accuracy improvements (v9) — and an honest accuracy check

**Key finding first.** A benchmark on 5 stocks (MSFT, AAPL, JPM,
RELIANCE.NS, TCS.NS; trained 2014 → 2024, tested on the last 15% of days
the models never saw) compared every setup against the **naive forecast**
"tomorrow's close = today's close":

| Stock | v8 ensemble vs naive (RMSE) | v9 ensemble vs naive (RMSE) |
|---|---|---|
| MSFT | 6.8% worse | 0.8% worse |
| AAPL | 28% worse | equal |
| JPM | 25% worse | 0.9% worse |
| RELIANCE.NS | 11% worse | equal |
| TCS.NS | 16% worse | 2.1% worse |

The old ensemble's R² of ~0.97 looked excellent, but the naive guess
scores *higher*. For daily prices, R² is ~0.97+ for almost any forecast,
because prices barely move day to day. Direction accuracy was ~50%, a coin flip.
v9 removes almost all of the error the models were *adding*. But no setup
tested reliably **beats** the naive forecast using price/volume data.
That matches the finance literature (markets are close to efficient
for next-day moves), and it's worth stating plainly in a report.

**What changed:**
- **Scale-free ("stationary") features** (`features.build_stationary_features`,
  39 features): ratios and % distances (price vs SMA 5/10/20/50/200,
  MA crossovers, 52-week high/low distance, relative volume, candle
  body/wick shape, gap, ATR %, ...) instead of raw price levels, which
  tree models can't extrapolate beyond. Used by the dashboard ensemble,
  and by `main.py` when `USE_STATIONARY_FEATURES = True` (default).
- **Regularized models, SVR removed from the dashboard ensemble**
  (`models_tabular.ENSEMBLE_MODEL_BUILDERS`): Ridge, a shallow Random
  Forest, and a constrained XGBoost. SVR was the least accurate model on
  every stock. `main.py` still compares all original models.
- **Shrinkage toward the naive forecast** (`dashboard_logic.ShrunkModel`):
  predictions are scaled by a factor α in [0, 1] fitted on validation data.
  α ≈ 0 means "no reliable signal found", so the forecast falls back to
  today's price instead of adding noise. It's shown in the dashboard as
  **Signal strength**. Expect α = 0 (and therefore mostly HOLD signals)
  for many stocks. That's the model being honest.
- **Bug fix — predictions used yesterday's data.** `build_features()`
  drops the newest day (its target isn't known yet), so the "next
  trading day" direction signal was actually predicting a day that had
  already happened, and each future-forecast step was one day stale.
  Fixed with `keep_latest=True` for all live predictions
  (dashboard, `predict_future.py`).
- **Fair ARIMA evaluation** (`model_arima.one_step_arima_forecast`):
  `main.py` used to score ARIMA on a single forecast spanning the whole
  test period (over a year ahead), while every other model predicts one day ahead.
  Its "−1500% vs naive" became −2% once scored on the same task.
- **Honest metrics everywhere.** `pipeline.evaluate_predictions()` now
  reports **Skill vs naive (%)** and **DirAcc (%)**. `main.py` adds a
  "Naive baseline" row. The dashboard's Model Insights tab shows skill
  vs naive, direction accuracy, and a held-out test for the direction
  classifiers against an "always predict UP" baseline.

**Things that were tried and did NOT help reliably** (worth a line in a
report): walk-forward retraining every quarter (±0.5%, mixed), scale-free
features for the direction classifiers (mixed across stocks), and
predicting 5-day instead of 1-day direction (mixed).

**If you want to go further**, the realistic sources of extra signal are
data the market hasn't fully priced in from OHLCV alone. Examples:
earnings surprises and dates, analyst revisions, historical news
sentiment (needs a paid archive), options-implied volatility, and
cross-sectional ranking across many stocks. Even then, expect a small
edge, not a big jump in accuracy.

## Stock search, candlestick reading & technical analysis (v8)

- **Search any stock by name** (`stock_search.py`). In the dashboard's
  sidebar, start typing a company name and real matching stocks are
  suggested live (Yahoo Finance's symbol search, so it covers NYSE,
  NASDAQ, NSE, BSE, LSE, ...). Pick one and the whole dashboard (models,
  charts, candlestick reading, news) reruns on that stock. Recently
  analyzed stocks get quick-switch buttons. A built-in list of popular
  US + Indian stocks is used as an offline fallback. The CLI scripts use
  the same search via a terminal prompt.
- **Company header**: full name, exchange, sector, market cap, P/E,
  beta, analyst consensus/target, and a company description. Prices use
  the stock's own currency (₹ for NSE stocks, $ for US, ...).
- **Candlestick chart reading** (`candlestick_patterns.py`, new
  🕯️ tab): detects 22 classic patterns (Doji, Hammer, Shooting Star,
  Engulfing, Harami, Piercing Line, Dark Cloud Cover, Morning/Evening
  Star, Three White Soldiers/Black Crows, Tweezers, Marubozu, ...), with
  a prior-trend filter (a hammer only counts after a decline), and turns
  the last few candles into a plain-English reading with a
  Bullish/Bearish/Neutral verdict. It also **backtests every pattern on
  the chosen stock's own history** (hit rate vs. base rate over the next
  5 days), so you can see which patterns actually had an edge on that
  stock instead of trusting textbook claims.
- **Candlestick feature in the ML models**: a `Candle_Bias_3d` feature
  (recent pattern bias x strength) is added to the direction classifiers.
- **Richer main chart**: timeframe selector (3M–5Y), volume panel,
  SMA 20/50/200, Bollinger Bands, pattern markers, and support /
  resistance lines, all toggleable.
- **Technical Analysis tab** (`technical_analysis.py`): an overall
  Strong Buy → Strong Sell rating that tallies 10 indicator votes (price
  vs SMA 20/50/200, golden/death cross, RSI, MACD, Bollinger %B,
  Stochastic, 1-month momentum, candlestick reading), a
  support/resistance table, and a Stochastic panel added under RSI/MACD.
- **Market context matched to the stock**: NSE/BSE stocks now use India
  VIX + NIFTY 50 (+ NIFTY IT / Bank / Pharma where relevant) instead of
  US VIX + S&P 500; US stocks get the sector ETF that matches their
  sector instead of always XLK.
- **Prices now match Google/Yahoo/your broker**: yfinance treats the
  end date as *exclusive*, so the dashboard (whose end date defaults to
  today) silently stopped at the previous trading day. `load_stock_data()`
  now treats `end` as inclusive. Prices also refresh every 15 minutes, the
  52-week high/low use intraday highs/lows (like every quote site), and
  during market hours the price is labelled "Live price (market open)"
  (Yahoo can lag the exchange by ~15 minutes). Prices and the chart load
  first; news and model training load after them.
- **Old / migrated listings handled automatically**: some search results
  (e.g. Indian SME-platform symbols like `OWAIS-SM.NS`) have almost no
  price history on Yahoo because the company moved to the main board
  (`OWAIS.NS`). `data_loader.load_stock_data()` now falls back to the
  related symbol that has the data (also tries .NS <-> .BO), and the
  dashboard tells you which listing it switched to. Stocks with under
  ~1.6 years of history get a clear "not enough history" message.
- **Bug fix**: stocks with zero-volume sessions (e.g. many NSE stocks) or
  indices used to crash training with "Input contains infinity".

> Install note: if `pip install streamlit-searchbox` downgrades
> `protobuf` and TensorFlow then fails to import, run
> `pip install "protobuf>=6.31.1" --no-deps` afterwards. Without
> `streamlit-searchbox` the dashboard still works, using a text box +
> "Did you mean…" dropdown instead of live suggestions.

## Real-world enhancements (beyond the coursework requirements)

These are additive -- your original price-prediction pipeline
(`main.py`, `ensemble.py`, `dashboard.py`'s price prediction) still
works exactly as before. This layer adds what actually matters if you
wanted to use this for real, alongside your predicted price, not
instead of it:

- **Market context features** (`data_loader.load_market_context()`):
  VIX (volatility index), S&P 500, and a sector ETF (default XLK for
  tech stocks) — a stock rarely moves independent of the broader
  market, and pure price/volume technical indicators can't capture that.
- **Volatility regime detection** (`features_advanced.py`): flags
  whether the last 20 days have been calmer or more turbulent than the
  stock's typical year, since models often behave differently in each.
- **Direction classification** (`models_classification.py`): Logistic
  Regression, Random Forest, and XGBoost trained to predict UP/DOWN
  instead of exact price — a fundamentally easier, more robust problem
  since it only needs the sign right, not the magnitude. Shown as a
  cross-check alongside price prediction, not a replacement.
- **Realistic transaction costs** (`backtest.py`): `commission_pct`
  and `slippage_pct` parameters (default 0, so existing behavior is
  unchanged unless you opt in) show how much of a strategy's paper
  profit survives real trading frictions — this matters a lot for
  high-trade-count strategies like the SVR example discussed earlier.
- **Live news sentiment** (`news_sentiment.py`): fetches current news
  headlines via yfinance's free built-in news feed and scores them with
  VADER sentiment analysis (boosted with a small finance-specific word
  list, since general VADER doesn't know words like "tumble" or
  "antitrust" are negative in a stock-news context). **Important
  limitation:** free news sources only give CURRENT headlines, not a
  historical archive, so this is shown as a LIVE overlay in the
  dashboard only — it is NOT and cannot be trained into the historical
  price/direction models, since there's nothing to backfill years of
  training data with for free. If you want true historical
  news-as-a-feature, that requires a paid/rate-limited provider (e.g.
  Alpha Vantage's NEWS_SENTIMENT endpoint has a free tier with some
  historical coverage and a public API key, if you want to extend this
  further).
- **Opening price prediction** (`dashboard_logic.py`): the dashboard
  now trains a SECOND small ensemble (same 4 tabular models) targeting
  the next day's OPENING price (expressed as the % overnight gap vs.
  today's close), alongside the original closing-price ensemble. Shown
  side by side in Tab 1's prediction results and in the future-forecast
  chart. This roughly doubles the dashboard's first-load training time,
  since it's training two full ensembles now instead of one.

**Run the combined script:**
```bash
python train_advanced.py
```
This runs your existing price models AND the new direction classifiers
AND a cost-aware backtest, all in one go, saving three CSVs.

**In the dashboard:** Tab 1 now has a "Direction Signal & Market
Regime" panel above the date picker, showing the blended direction
classifier confidence and current volatility regime as a live
cross-check alongside your price prediction.

**Honest context worth keeping in your report:** for a liquid,
large-cap stock, 50% direction accuracy is a coin flip; professional
quant funds typically achieve only 51-55% on next-day direction for
stocks like this. Treat any number far above that with suspicion
(likely overfitting) rather than excitement — the ceiling here is a
property of efficient markets, not a limitation of this code.

## Interactive dashboard

Run the full dashboard with:
```bash
streamlit run dashboard.py
```
This opens a browser tab with:
- **A stock search box** with live name suggestions, a company header,
  and **an interactive candlestick chart** (timeframes, volume, moving
  averages, Bollinger Bands, pattern markers, support/resistance).
- **Tab 1 — Predict a Date:** pick any date and get the ensemble's
  predicted closing price, a color-coded 🟢BUY / 🔴SELL / ⚪HOLD badge,
  and an interactive chart showing the prediction in context (actual
  price history for historical dates, or the full recursive forecast
  path for future dates).
- **Tab 2 — Best Buy/Sell Window:** pick a lookahead horizon and get
  the model's best predicted buy point and best predicted sell point
  after it, plotted with markers on the forecast path.
- **Tab 3 — Model Insights:** a bar chart of each model's weight in the
  live ensemble, plus a button to compute the ensemble's combined
  accuracy (MAE/RMSE/MAPE/R²) on a proper held-out test set — the same
  number `ensemble_accuracy.py` reports, shown live in the UI.

The first run takes a minute or two (training all 5 models); after
that, results are cached so trying different dates is fast. All the
prediction logic lives in `dashboard_logic.py`; `dashboard.py` is just
the UI wiring (Streamlit + Plotly) on top of it.

**Caveat worth repeating in your report:** dates beyond the historical
dataset are forecast recursively (each day builds on the previous
day's *prediction*, not real data), so reliability drops the further
ahead you ask — this is flagged directly in the dashboard's sidebar
and warning messages too.

## File structure

| File | Purpose |
|---|---|
| `data_loader.py` | Downloads OHLCV data via yfinance |
| `features.py` | Builds technical indicators (RSI, MACD, Bollinger Bands, moving averages, lagged returns) + the prediction target |
| `pipeline.py` | Chronological train/val/test split + shared evaluation metrics — **used identically by every model** so the comparison is fair |
| `models_tabular.py` | Linear Regression, Random Forest, SVR, XGBoost |
| `model_arima.py` | ARIMA (univariate baseline, Close price only, auto order selection via AIC) |
| `models_sequential.py` | LSTM, GRU, Transformer + sequence-windowing helper, with EarlyStopping |
| `backtest.py` | Buy/Sell/Hold trading simulation + Buy-and-Hold baseline |
| `main.py` | Orchestrates everything end-to-end, both target modes (price + return), all 14 models + ARIMA |
| `plot_results.py` | Generates `actual_vs_predicted.png` and `portfolio_comparison.png` using one fast tabular model (default: Linear Regression) |
| `ensemble.py` | Combines predictions from all models into ONE blended forecast and ONE trading strategy backtest |
| `predict_future.py` | Predicts closing price on specific future dates beyond your historical dataset, via recursive forecasting |
| `dashboard_logic.py` | Core prediction/ensemble/signal logic behind the dashboard (trainable, testable, no UI code) |
| `dashboard.py` | Interactive Streamlit dashboard — pick a date, get a price + Buy/Sell/Hold signal; scan for the best buy/sell window |
| `models_classification.py` | Direction (up/down) classifiers — Logistic Regression, Random Forest, XGBoost |
| `features_advanced.py` | Adds market context, volatility regime, and direction target on top of the base features |
| `train_advanced.py` | Standalone script: price prediction + direction classification + cost-aware backtest, all together |
| `news_sentiment.py` | Live news sentiment via yfinance's free news feed + VADER (finance-boosted) sentiment scoring |
| `ensemble_accuracy.py` | Reports the combined ensemble's accuracy on a proper held-out test set |
| `tune_hyperparameters.py` | Time-series-aware hyperparameter search for Random Forest, SVR, XGBoost |
| `stock_search.py` | Search stocks by company name (live Yahoo suggestions + offline list), terminal prompt for CLI scripts, company profile |
| `candlestick_patterns.py` | Detects 22 candlestick patterns, plain-English chart reading, per-stock pattern reliability backtest |
| `ipo_analyzer.py` | IPO data (NSE + IPOWatch), GMP reliability, apply/avoid scorecard |
| `ipo_page.py` | IPO Center page: search, live GMP board, per-IPO analysis, recent listings, GMP reliability |
| `model_smoothing.py` | Simple Exponential Smoothing with fixed alpha = 0.3: forecasting model, engine features, chart overlay |
| `research_analyst.py` | Research analyst: fundamentals, financial trends, valuation, price history, ownership -> pillar scores, verdict and written report |
| `forecast_engine.py` | Price forecasting engine: all feature groups, walk-forward group selection, direct 1/5/10/21/63-day models, 80% ranges, held-out evaluation |
| `ui_theme.py` | Dashboard design system: colors, CSS, Plotly chart theme, HTML building blocks (hero, KPI cards, snapshot cards) |
| `news_history.py` | Historical news features for the models: GDELT (free), Alpha Vantage (optional key), own daily collection; cached in `news_cache/` |
| `collect_news.py` | Daily news collector for a watchlist (schedule it to build your own news history) |
| `technical_analysis.py` | Support/resistance levels, technical indicator scorecard, extra chart indicators (SMA 200, Stochastic, ATR) |

## Why some models show 0 trades

The trading strategy only buys/sells when a model's predicted next-day
% change exceeds ±1%. Day-to-day stock moves are usually small, so an
accurate model (whose predictions closely track reality) will rarely
cross that threshold — while a less accurate, more volatile model
(e.g. SVR) triggers far more trades, for better or worse. This is a
genuine, worth-discussing finding: forecasting accuracy and a fixed
% threshold trading strategy don't automatically align. Two things
worth trying if you want more trades from your best models: lower the
threshold (e.g. 0.3–0.5%), or scale the threshold to each model's own
prediction volatility instead of using one fixed number for everyone.

## Combining all models into one (ensemble)

Run `python ensemble.py` to combine predictions from Linear Regression,
Random Forest, SVR, XGBoost, and ARIMA into a single blended forecast
and a single trading-strategy backtest, instead of treating every model
separately. Two weighting options (set `WEIGHTING` in the CONFIG block):
- `"equal"` — simple average across all included models.
- `"inverse_rmse"` (default) — models that were more accurate on the
  validation set get proportionally more weight in the blend.

Set `INCLUDE_SEQUENTIAL_MODELS = True` in `ensemble.py` to also fold in
LSTM/GRU/Transformer (slower to run, since they need real training time).

## Predicting specific future dates

`predict_future.py` predicts the stock's closing price on dates you
choose that fall *after* your historical dataset ends. Since future
Open/High/Low/Volume don't exist yet, it forecasts recursively — one
day at a time, feeding each prediction back in as if it were real, then
predicting the next day from that extended series. This means accuracy
degrades the further into the future you ask for (a common, well-known
limitation of multi-step forecasting, not a flaw in this code — worth
mentioning explicitly if you use this in your report). Edit
`FUTURE_DATES` in the CONFIG block, then run `python predict_future.py`.

## Design notes worth mentioning in your report

- **Chronological splitting only.** No random shuffling — this
  prevents future data from leaking into training, which is a common
  mistake that makes results look artificially good.
- **Same features, same split, same metrics for every model.** This is
  what makes the comparison scientifically valid rather than just a
  set of separate experiments.
- **Every model (except ARIMA) is trained TWICE: once to predict raw
  next-day price, once to predict next-day % return.** This matters
  because tree-based models (Random Forest, XGBoost) and SVR cannot
  extrapolate beyond the price range seen in training. On a stock that
  trended strongly upward (like MSFT), the test period contains prices
  higher than anything the model trained on, so "price mode" alone
  makes these models look artificially terrible (even negative R²).
  "Return mode" fixes this since % returns don't have the same
  extrapolation problem. Return predictions are converted back into
  price terms (`predicted_price = today_close * (1 + predicted_return)`)
  so every model's final metrics and backtest are on the same scale
  and directly comparable. **Reporting this comparison explicitly is a
  legitimate, interesting finding for your write-up** — it shows you
  understand a real limitation of ML models on trending financial data.
- **ARIMA uses an automatic order search** (`select_best_order` in
  `model_arima.py`) instead of a fixed `(p,d,q)`, picking whichever
  combination minimizes AIC on the training data — a lightweight
  version of what `auto_arima` does, without an extra dependency.
- **LSTM/GRU/Transformer use EarlyStopping** (`models_sequential.py`)
  with a generous epoch budget (100) — training automatically stops
  once validation loss stops improving, and the best-performing
  weights are restored. This avoids both underfitting (too few epochs)
  and overfitting (training past the point of diminishing returns).
- **XGBoost uses early stopping via a validation set** rather than a
  fixed number of estimators, for the same reason.
- **Trading strategy is a simple threshold rule** (buy if predicted
  next-day return > +1%, sell if < -1%, else hold) — intentionally
  simple so it's easy to explain and defend, with thresholds and
  starting capital adjustable in `backtest.py` / `main.py`.
- **Transformer is intentionally small.** Large transformers tend to
  overfit on a dataset of a few thousand daily rows; a compact
  encoder is more appropriate here and is easy to justify.
- **Transaction costs are not modeled** — mention this as a
  simplification in your report's limitations section.
- **Results are for academic backtesting only**, not investment advice
  — state this explicitly in your report.

## Suggested extensions if you have time left

- Tune ARIMA's `(p,d,q)` order via AIC/BIC grid search instead of the
  fixed `(5,1,0)` default.
- Try `rolling_arima_forecast()` in `model_arima.py` (walk-forward
  validation) instead of the faster one-shot forecast — more realistic
  but much slower.
- Add a second stock and repeat the comparison, if your assignment
  wants a multi-stock analysis.
- Increase `EPOCHS` and tune LSTM/GRU/Transformer hyperparameters
  (units, layers, dropout) once the base pipeline works end-to-end.
