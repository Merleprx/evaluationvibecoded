# Residual Income Valuation App (Phase III B)

An interactive Streamlit app. You type a stock ticker (NKE, AAPL, MSFT, ...), and the app does the following:

- pulls the financial and market data from Yahoo Finance (yfinance)
- forecasts revenue and net income, then values the equity with a residual income model
- compares the result with trading comparables (EV/Revenue, EV/EBITDA, P/E, P/Book) and the current share price
- shows everything in tables, charts and a football field

## Files

| File | What it is |
|---|---|
| `app.py` | The app (user interface, Yahoo Finance data, charts) |
| `valuation.py` | The valuation formulas (residual income, sensitivity, comps) |
| `requirements.txt` | The Python packages the app needs |
| `run_app.command` | Mac shortcut: double-click to install everything and start the app |

## Run it on a Mac (beginner steps)

1. **Put the folder somewhere easy**, for example on the Desktop: `Desktop/nike_valuation_app`.
2. **Check Python.** Open *Terminal* (Cmd + Space, type "Terminal") and run:
   ```bash
   python3 --version
   ```
   If you see `Python 3.10` or higher, you're set. Otherwise, install Python from https://www.python.org/downloads/. You can also use Anaconda if you already have it.
3. **Start the app.** There are two ways:
   - **Easiest:** double-click `run_app.command`. If macOS blocks it, right-click it, choose **Open**, then **Open** again. The first run installs the packages (1-3 minutes).
   - **In Terminal:**
     ```bash
     cd ~/Desktop/nike_valuation_app
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     streamlit run app.py
     ```
4. Your browser opens at **http://localhost:8501**. To stop the app, press **Ctrl + C** in Terminal.
5. **Next time:** run `source .venv/bin/activate` and then `streamlit run app.py` from the folder, or double-click `run_app.command` again.

On Windows, use `python` instead of `python3`, and `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## How to use it

- **Sidebar 1 - Company.** Enter a ticker. For NKE, the switch **"Use course base case"** loads the problem-set assumptions. The app then shows **$36.97 per share**, the same value as the Excel model.
- **Sidebar 2 - Assumptions.** These are all editable:
  - Revenue growth
  - Net income margin
  - Payout ratio
  - Cost of equity (from CAPM, or typed in directly)
  - Terminal growth
  - Forecast period (3-10 years)

  Open **"Year-by-year growth and margin"** to set a different value for each year.
- **Sidebar 3 - Base data.** Revenue, book value, shares and price are filled in from Yahoo. Change any value that is missing or looks wrong.
- **Sidebar 4 - Trading comparables.** Enter peer tickers, for example `ADDYY, LULU, DECK, ONON, CROX, UAA` for Nike.

**Tabs:**

| Tab | What it shows |
|---|---|
| Forecast | Forecast table and charts |
| Valuation | Bridge from book value to equity value, cost of equity x terminal growth heatmap, driver sensitivity |
| Trading comps | Peer multiples and the implied share prices |
| Football field | All valuation ranges against the market price |
| Market vs. value | 1-year price chart, and the terminal growth the current market price implies |
| History & notes | Reported history, CSV download, method |

## Method

| Step | Formula |
|---|---|
| Revenue | Revenue(t) = Revenue(t-1) x (1 + growth) |
| Net income | Net income = Revenue x net margin |
| Dividends | Dividends = Net income x payout |
| Book value | BV(t) = BV(t-1) + NI - Dividends |
| Residual income | RI = NI - cost of equity x beginning BV |
| Terminal value | TV = RI(last year) x (1 + g) / (ke - g) |
| Equity value | BV(0) + PV(RI) + PV(TV) |
| Value per share | Equity value / shares |

## Put it online and share a link (GitHub + Streamlit Community Cloud, free)

**1. Create a GitHub account and an empty repository**
- Sign up at https://github.com, then click **New** (or go to https://github.com/new).
- Name it e.g. `nike-valuation-app`, choose **Public**, and click **Create repository**.

**2. Upload the files.** Pick one of the two ways.

*In the browser, no Git needed:*
1. On the new repository page, click **uploading an existing file**.
2. Drag in `app.py`, `valuation.py`, `requirements.txt`, `README.md` and `.gitignore`, then click **Commit changes**.
3. Add the theme file. The `.streamlit` folder is hidden in Finder; press **Cmd + Shift + .** to show it, or create the file on GitHub:
   - Click **Add file > Create new file**.
   - Type the name `.streamlit/config.toml` (the slash creates the folder).
   - Paste the contents of the local `config.toml`, then click **Commit changes**.

*Or in Terminal, from the app folder:*
```bash
git init
git add .
git commit -m "Residual income valuation app"
git branch -M main
git remote add origin https://github.com/<your-username>/nike-valuation-app.git
git push -u origin main
```
GitHub asks for a personal access token instead of your password. Create one under Settings > Developer settings.

**3. Deploy on Streamlit Community Cloud**
1. Go to https://share.streamlit.io and sign in with GitHub. Allow access when asked.
2. Click **Create app**, then **Yup, I have an app**.
3. Choose your repository, branch `main`, and main file `app.py`.
4. Optionally set the app URL, e.g. `nike-valuation`.
5. Under **Advanced settings**, pick Python **3.12**, then click **Deploy**.
6. After a few minutes the app is live at `https://<your-subdomain>.streamlit.app`.
7. Send that link to your professor. A public repository means anyone with the link can open the app.
8. Every new commit to GitHub updates the app automatically.

**If the online app says Yahoo Finance did not return data:** Yahoo sometimes rate-limits shared cloud servers. Wait a few minutes and reload. With NKE and the course base case on, the valuation still works from the built-in course data.

## Good to know

- Yahoo Finance data is free and unofficial. It can be missing, delayed or rate-limited ("Too Many Requests"). Wait a minute and refresh, or type the numbers into the sidebar.
- Some companies report in a currency different from their share price (e.g. ADDYY). The comps use Yahoo's own ratios, which handle this. The app warns you if it affects the company you are valuing.
