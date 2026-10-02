# Updating the deployed site

Your app: **https://stocksense0906.streamlit.app/**

Streamlit Community Cloud does not take uploads. It watches a **GitHub branch**
and redeploys whenever you push to it. So updating the site means pushing these
files to that repo.

---

## The short version

```bash
cd <your-local-clone-of-the-repo>
# copy everything from this folder over the repo, then:
git add -A
git commit -m "Pooled per-asset-class models, FX leak fix, IPO Center fix"
git push
```

Streamlit sees the push and rebuilds in 2-5 minutes. Watch it from
**Manage app → Logs** in the bottom-right of your site.

---

## Step by step

### 1. Get the repo locally

If you already have the clone you deployed from, skip to step 2.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Not sure which repo it is? Open the app, click **Manage app** (bottom right),
and the source repo and branch are shown at the top.

### 2. Copy these files in

Copy the whole contents of this folder over your clone, replacing what is
there. Keep your own `README.md` and anything else you added.

### 3. Check what git sees

```bash
git status
```

You should see modified `.py` files, a new `requirements.txt`, and two files
under `news_cache/`. You should **not** see `fo_cache/`, `panel_all_assets.parquet`
or `kite_credentials.json` — the included `.gitignore` keeps them out.

If `git status` is huge (hundreds of parquet files), the `.gitignore` did not
land. Fix that before committing — those directories are ~750 MB.

### 4. Commit and push

```bash
git add -A
git commit -m "Pooled per-asset-class models, FX leak fix, IPO Center fix"
git push
```

### 5. Watch the rebuild

On the site: **Manage app → Logs**. A healthy deploy ends with a line about the
app being served. If it fails, the log says why — see Troubleshooting below.

---

## What changed in this update

| Area | Change |
|---|---|
| Forecast models | Trained across 129 instruments instead of one symbol at a time, grouped into four pools by asset class |
| Data integrity | New check that detects feeds whose daily high/low cannot be trusted, and disables the affected features |
| IPO Center | "Recent listings" and "How reliable is GMP" were blank because a data source moved; both now work |
| Dependencies | `tensorflow-cpu` removed (see below) |

### Two new files

`news_cache/pooled_model.pkl` and `news_cache/pooled_oof.pkl` are the trained
models, about 26 MB together. **They must be committed** — the dashboard loads
them at startup. Without them the app still runs, it just silently drops the
pooled features and the forecasts get weaker.

26 MB is fine for GitHub (the per-file limit is 100 MB). You do not need Git LFS.

---

## Why `tensorflow-cpu` was removed

It was in your old `requirements.txt`, but only `models_sequential.py` imports
it, and that is reached from the `main.py` command line tool — never from the
dashboard. On Streamlit's free tier it costs around 250 MB of build time and a
large share of the 1 GB memory limit, for code the website never executes. That
combination is a common cause of an app that builds slowly and then dies on
boot.

Nothing on the site loses any feature. To use the LSTM/GRU models locally:

```bash
pip install tensorflow-cpu
```

---

## Secrets: do not commit them

If you ever connect the Zerodha Kite API, the credentials go in Streamlit's own
secrets store, never in the repo:

**Manage app → Settings → Secrets**, then paste:

```toml
KITE_API_KEY = "..."
KITE_API_SECRET = "..."
KITE_ACCESS_TOKEN = "..."
```

The code reads them from the environment, so this works without any change.
`kite_credentials.json` is in `.gitignore` and must stay there — a public repo
means a public key.

---

## Troubleshooting

**Build fails on a package**
Read the log for the package name. Pin a version in `requirements.txt`, e.g.
`numpy==1.26.4`, and push again.

**App boots then crashes with no clear error**
Almost always memory. The free tier gives 1 GB. Analysing a symbol trains five
forecast horizons and can approach that. Reduce the default history window in
the sidebar, or upgrade the instance.

**First load is very slow**
Expected. The first analysis of a symbol trains its models — 1-3 minutes
locally, longer on shared cloud CPU. Results are cached afterwards, so the
second visit is fast. The app sleeps after inactivity, and the next visitor
pays that cost again.

**IPO page shows no data**
The upstream sources (NSE, IPOWatch) sometimes block cloud IP ranges even when
they work from your laptop. The page degrades to a warning rather than crashing.

**A panel that used to work goes blank**
A data source changed its page structure. This has happened before. Run
`python audit.py` locally — it checks that each source still returns what the
code expects, and exits with an error if not.

---

## Keeping it fresh

Retrain the pooled models about monthly, then commit the two refreshed pickles:

```bash
python pooled_model.py
git add news_cache/pooled_model.pkl news_cache/pooled_oof.pkl
git commit -m "Retrain pooled models"
git push
```

Run `python audit.py` after any code change. It exits non-zero if a data
integrity rule is broken, which makes it usable as a pre-push check.
