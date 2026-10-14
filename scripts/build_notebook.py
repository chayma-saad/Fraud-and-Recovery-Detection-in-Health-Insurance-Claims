"""Generates notebooks/data_processing.ipynb from scratch as an nbformat JSON
structure, then the notebook is executed separately via nbconvert so every
output/figure is baked in and visible without the user re-running anything."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ==========================================================================
md(r"""# Insurance Claims Data Processing Pipeline

This notebook walks through the same pipeline as `scripts/process_data.py`, step by
step, with explanations and **charts at every stage** so you can see what the raw
Excel exports actually contain and how each transformation changes the data.

**Pipeline stages:**
1. Locate and load the raw Excel workbooks (claims + beneficiaries)
2. Clean and standardize columns (IDs, dates, financial amounts)
3. Apply business logic (status definitions, valid/excluded split, beneficiary merge,
   post-termination flag, recovery sub-table)
4. Export processed CSVs
5. Explore the result visually (volumes, status mix, money flows, trends, demographics)
""")

# ==========================================================================
md("## 0. Setup\n\nStandard imports plus a small chart style block (colors picked from a colorblind-safe categorical palette).")

code(r"""
import re
import shutil
import tempfile
import zipfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

pd.set_option("display.max_columns", 60)

# --- Chart styling: fixed-order categorical palette (colorblind-safe) ---
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10.5,
    "figure.dpi": 100,
})


def fmt_thousands(ax, axis="y"):
    formatter = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)
""")

# ==========================================================================
md(r"""## 1. Paths & metadata

Where the raw workbooks live, and the reference dictionaries used later (status
definitions, financial column list, column-name aliases, the anchor text used to
locate the real header row under the title rows on each sheet).""")

code(r"""
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

DATA_DIR = BASE_DIR / "data"
RAW_CLAIMS_DIR = DATA_DIR / "raw" / "utilization"
RAW_BENEF_DIR = DATA_DIR / "raw" / "beneficiaries"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("RAW_CLAIMS_DIR:", RAW_CLAIMS_DIR.resolve())
print("  contains:", sorted(f.name for f in RAW_CLAIMS_DIR.iterdir() if f.is_file()))
print("RAW_BENEF_DIR:", RAW_BENEF_DIR.resolve())
print("  contains:", sorted(f.name for f in RAW_BENEF_DIR.iterdir() if f.is_file()))
""")

code(r"""
STATUS_DEFINITIONS = {
    "Initial": "Authorization entered in NAS MAIN system or Direct Claim in initial processing stage",
    "Authorized": "Authorization issued in JET / E-Claim only",
    "Validated": "A Non-E-Claim batched and entered in the system before processing",
    "Pending": "Claim entered and decision pended for different reasons",
    "Processed": "Claims entered and decision made",
    "Closed": "Decision checked by system for limits/sublimits; ready for payment",
    "Not Closed & Pended (Partial)": "Internal status (e.g., Maternity); pending further adjudication regarding recovery",
    "PO Issued": "All claims previously closed included in Payment Orders for settlement",
    "Settled": "All claims for which payment has been dispatched to provider/beneficiary",
    "Rejected": "All authorizations which were rejected",
    "Void": "An authorization issued but not utilized",
    "Pending Validation": "Requests finished by provider and sent to Claim Center for approval",
    "Required Further Details": "Requests sent back to provider for more information",
    "Request Forwarded": "Requests forwarded between users but not finalized",
    "Replied Further Details": "Requests responded back by provider",
    "Pending With Payer": "Requests escalated to Payer",
    "Pending For 2nd Opinion": "Requests pended for second opinion",
}

ID_COL_CANDIDATES = ["BENEFICIARYIDENTIFIER", "BENEFICIARYID"]

FINANCIAL_COLS = [
    "PAYERAMOUNTCLAIMED", "BENEFICIARYAMOUNTCLAIMED", "TOTALAMOUNTCLAIMED",
    "TOTALPAYABLE", "PAYERSHARE", "BENEFICIARYSHAREAPPROVED",
    "DEDUCTIONAMOUNT", "RECOVERY/EX-GRATIA",
]

COLUMN_ALIASES = {"RECOVERY": "RECOVERY/EX-GRATIA"}

HEADER_ANCHORS = ("CONTRACT", "BENEFICIARYID")
MAX_TITLE_ROWS_TO_SCAN = 15
SERVICE_DATE_COL = "TXNDATE"
POLICY_EXPIRY_COL = "EXPIRYDATE"
""")

# ==========================================================================
md(r"""## 2. Helper functions

These mirror `process_data.py` exactly:
- **`strip_defined_names`** works around an openpyxl crash on legacy workbooks that
  have a corrupted "defined name" (named range) — it makes a cleaned temp copy.
- **`detect_header_row`** scans the first rows of a sheet for the real header, since
  the raw exports have title rows above the actual column headers.
- **`load_workbook_sheets`** opens every `.xlsx` in a folder and loads every sheet
  inside it, tagging each row with which sheet/file it came from.""")

code(r"""
_TEMP_CLEANED_FILES = []


def strip_defined_names(src_path):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(tmp_fd)
    shutil.copy(src_path, tmp_path)
    try:
        with zipfile.ZipFile(tmp_path, "r") as zin:
            contents = {n: zin.read(n) for n in zin.namelist()}
        if "xl/workbook.xml" in contents:
            xml = contents["xl/workbook.xml"].decode("utf-8", errors="replace")
            cleaned_xml = re.sub(r"<definedNames>.*?</definedNames>", "", xml, flags=re.DOTALL)
            if cleaned_xml != xml:
                print(f"    -> '{Path(src_path).name}' had defined names that can crash openpyxl on read; stripped them from a temp copy.")
                contents["xl/workbook.xml"] = cleaned_xml.encode("utf-8")
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
                    for name, data in contents.items():
                        zout.writestr(name, data)
    except Exception:
        pass
    _TEMP_CLEANED_FILES.append(tmp_path)
    return Path(tmp_path)


def get_readable_path(path):
    return strip_defined_names(path)


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_header_row(raw_df, anchors=HEADER_ANCHORS, max_rows=MAX_TITLE_ROWS_TO_SCAN):
    scan_limit = min(max_rows, len(raw_df))
    for i in range(scan_limit):
        row_cells = raw_df.iloc[i].astype(str).str.strip().str.upper().tolist()
        if all(any(anchor in cell for cell in row_cells) for anchor in anchors):
            return i
    return None


def read_excel_robust(file, sheet_name, header=None, nrows=None):
    try:
        return pd.read_excel(file, sheet_name=sheet_name, header=header, nrows=nrows, engine="calamine")
    except Exception:
        return pd.read_excel(file, sheet_name=sheet_name, header=header, nrows=nrows, engine="openpyxl")


def load_tagged_sheet(file, sheet_name, tag_col, display_name=None):
    raw_preview = read_excel_robust(file, sheet_name, header=None, nrows=MAX_TITLE_ROWS_TO_SCAN)
    header_row = detect_header_row(raw_preview)
    if header_row is None:
        print(f"  WARNING: could not find a header row in sheet '{sheet_name}' of {display_name or file.name} — skipping this sheet.")
        return None
    df = read_excel_robust(file, sheet_name, header=header_row)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    df[tag_col] = sheet_name
    df["source_file"] = display_name or file.name
    print(f"  Loaded sheet '{sheet_name}' from {display_name or file.name}: {len(df):,} rows (header found at row {header_row + 1})")
    return df


def load_workbook_sheets(directory, tag_col):
    files = sorted(
        {f.resolve() for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".xlsx"},
        key=lambda f: f.name,
    )
    frames = []
    for file in files:
        try:
            readable_file = get_readable_path(file)
            try:
                sheet_names = pd.ExcelFile(readable_file, engine="calamine").sheet_names
            except Exception:
                sheet_names = pd.ExcelFile(readable_file, engine="openpyxl").sheet_names
        except Exception as e:
            print(f"  Error opening {file.name}: {e}")
            continue
        for sheet_name in sheet_names:
            try:
                df_sheet = load_tagged_sheet(readable_file, sheet_name, tag_col, display_name=file.name)
                if df_sheet is not None and not df_sheet.empty:
                    frames.append(df_sheet)
            except Exception as e:
                print(f"  Error reading sheet '{sheet_name}' in {file.name}: {e}")
    return frames, len(files)
""")

# ==========================================================================
md("## 3. Load & combine claims data\n\nEvery sheet of every workbook in `data/raw/utilization/` is loaded and tagged with `REPORT_TYPE` (the sheet name it came from), then concatenated into one DataFrame.")

code(r"""
claim_frames, n_claim_files = load_workbook_sheets(RAW_CLAIMS_DIR, tag_col="REPORT_TYPE")
print(f"\nFound {n_claim_files} claims workbook(s)")

if not claim_frames:
    raise FileNotFoundError("No valid claims data loaded. Please check data/raw/utilization.")

df_utilization = pd.concat(claim_frames, ignore_index=True, sort=False)
print(f"Combined claims dataset: {len(df_utilization):,} rows across report types: "
      f"{sorted(df_utilization['REPORT_TYPE'].unique())}")
df_utilization.head()
""")

md("**Chart — rows loaded per source workbook.** Sanity check that every raw file actually contributed rows.")

code(r"""
counts_by_file = df_utilization["source_file"].value_counts()

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(counts_by_file.index[::-1], counts_by_file.values[::-1], color=CATEGORICAL[0], height=0.6)
ax.set_xlabel("Rows loaded")
ax.set_title("Claims rows loaded per source workbook", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="x")
for bar, v in zip(bars, counts_by_file.values[::-1]):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {v:,}", va="center", color=INK_SECONDARY, fontsize=9)
plt.tight_layout()
plt.show()
""")

# ==========================================================================
md("## 4. Load & combine beneficiary data\n\nSame process, applied to `data/raw/beneficiaries/`, tagged with `BENEF_SHEET` instead.")

code(r"""
benef_frames, n_benef_files = load_workbook_sheets(RAW_BENEF_DIR, tag_col="BENEF_SHEET")
print(f"\nFound {n_benef_files} beneficiary workbook(s)")

df_beneficiaries = pd.concat(benef_frames, ignore_index=True, sort=False) if benef_frames else pd.DataFrame()
print(f"Combined beneficiaries dataset: {len(df_beneficiaries):,} rows")
df_beneficiaries.head()
""")

# ==========================================================================
md("## 5. Data cleaning & standardization\n\nID columns are trimmed/uppercased for a reliable join key; date columns are parsed; financial columns are coerced to numeric with a `_WAS_MISSING` flag recorded **before** filling NaNs with 0.0, so a genuine \\$0 claim stays distinguishable from a blank/unparseable source cell.")

code(r"""
util_id_col = find_column(df_utilization, ID_COL_CANDIDATES)
if util_id_col:
    df_utilization[util_id_col] = df_utilization[util_id_col].astype(str).str.strip().str.upper()
    print(f"Claims ID column: '{util_id_col}'")

benef_id_col = find_column(df_beneficiaries, ID_COL_CANDIDATES) if not df_beneficiaries.empty else None
if benef_id_col:
    df_beneficiaries[benef_id_col] = df_beneficiaries[benef_id_col].astype(str).str.strip().str.upper()
    df_beneficiaries = df_beneficiaries.drop_duplicates(subset=[benef_id_col])
    print(f"Beneficiary ID column: '{benef_id_col}'")

date_cols_util = [c for c in df_utilization.columns if "DATE" in c.upper() or c.upper() == "DOB"]
for col in date_cols_util:
    df_utilization[col] = pd.to_datetime(df_utilization[col], errors="coerce")

if not df_beneficiaries.empty:
    date_cols_benef = [c for c in df_beneficiaries.columns if "DATE" in c.upper() or c.upper() == "DOB"]
    for col in date_cols_benef:
        df_beneficiaries[col] = pd.to_datetime(df_beneficiaries[col], errors="coerce")

missing_summary = {}
for col in FINANCIAL_COLS:
    if col in df_utilization.columns:
        parsed = pd.to_numeric(df_utilization[col], errors="coerce")
        n_missing = int(parsed.isna().sum())
        missing_summary[col] = n_missing
        if n_missing:
            df_utilization[col + "_WAS_MISSING"] = parsed.isna()
        df_utilization[col] = parsed.fillna(0.0)

if "STATUS" in df_utilization.columns:
    df_utilization["STATUS"] = df_utilization["STATUS"].astype(str).str.strip()

missing_summary
""")

md("**Chart — missing/unparseable values filled per financial column.**")

code(r"""
ms = pd.Series(missing_summary).sort_values(ascending=False)
ms = ms[ms > 0]

if len(ms):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ms.index, ms.values, color=CATEGORICAL[1], width=0.6)
    ax.set_ylabel("Rows filled with 0.0")
    ax.set_title("Missing/unparseable values by financial column", color=INK, fontsize=12, loc="left", pad=12)
    fmt_thousands(ax, axis="y")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No missing financial values found.")
""")

# ==========================================================================
md("## 6. Business logic\n\n**A.** Map each `STATUS` to its plain-English definition.\n\n**B.** Split off `Rejected`/`Void` claims into their own frame instead of silently discarding them.\n\n**C.** Left-join the valid claims onto beneficiary data.\n\n**D.** Flag claims serviced *after* the policy's expiry date.\n\n**E.** Pull out the recovery/ex-gratia sub-table.")

code(r"""
if "STATUS" in df_utilization.columns:
    df_utilization["STATUS_DESCRIPTION"] = df_utilization["STATUS"].map(STATUS_DEFINITIONS).fillna("Unknown / Unmapped Status")

if "STATUS" in df_utilization.columns:
    is_excluded = df_utilization["STATUS"].isin(["Rejected", "Void"])
    df_valid_claims = df_utilization[~is_excluded].copy()
    df_excluded_claims = df_utilization[is_excluded].copy()
else:
    df_valid_claims = df_utilization.copy()
    df_excluded_claims = pd.DataFrame()

print(f"Valid claims: {len(df_valid_claims):,}  |  Excluded (Rejected/Void): {len(df_excluded_claims):,}")

if not df_beneficiaries.empty and util_id_col and benef_id_col and util_id_col in df_valid_claims.columns:
    df_final = pd.merge(
        df_valid_claims, df_beneficiaries,
        left_on=util_id_col, right_on=benef_id_col,
        how="left", suffixes=("", "_beneficiary"),
    )
else:
    df_final = df_valid_claims
    print("Skipped beneficiary merge (missing ID column on one or both sides).")

if SERVICE_DATE_COL in df_final.columns and POLICY_EXPIRY_COL in df_final.columns:
    df_final["IS_POST_TERMINATION"] = df_final[SERVICE_DATE_COL] > df_final[POLICY_EXPIRY_COL]
    n_flagged = int(df_final["IS_POST_TERMINATION"].sum())
    print(f"Post-termination flag: {n_flagged:,} rows flagged")
else:
    df_final["IS_POST_TERMINATION"] = False

df_recovery = df_final[df_final["RECOVERY/EX-GRATIA"] > 0].copy() if "RECOVERY/EX-GRATIA" in df_final.columns else pd.DataFrame()

print(f"\nFinal merged dataset: {len(df_final):,} rows, {df_final.shape[1]} columns")
print(f"Recovery/ex-gratia rows: {len(df_recovery):,}")
df_final.head()
""")

# ==========================================================================
md("## 7. Export processed CSVs\n\nWrites the same files as the script: the combined dataset, one CSV per report type, the recovery sub-table, and the excluded (Rejected/Void) claims.")

code(r"""
final_output = PROCESSED_DIR / "consolidated_claims_all_reports.csv"
df_final.to_csv(final_output, index=False)
print(f"Saved: {final_output.name}")

for report_type, df_group in df_final.groupby("REPORT_TYPE"):
    safe_name = "".join(c if c.isalnum() else "_" for c in str(report_type)).strip("_").lower()
    per_type_output = PROCESSED_DIR / f"claims_{safe_name}.csv"
    df_group.to_csv(per_type_output, index=False)
    print(f"Saved: {per_type_output.name} ({len(df_group):,} rows)")

if not df_recovery.empty:
    recovery_output = PROCESSED_DIR / "recovery_ex_gratia_claims.csv"
    df_recovery.to_csv(recovery_output, index=False)
    print(f"Saved: {recovery_output.name}")

if not df_excluded_claims.empty:
    excluded_output = PROCESSED_DIR / "excluded_rejected_void_claims.csv"
    df_excluded_claims.to_csv(excluded_output, index=False)
    print(f"Saved: {excluded_output.name}")

for tmp_path in _TEMP_CLEANED_FILES:
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

print("\nPipeline execution finished successfully!")
""")

# ==========================================================================
md(r"""## 8. Exploratory charts

A visual look at the processed dataset: volumes, claim status mix, money flows,
trends over time, and demographics.""")

md("### 8.1 Report type mix\n\nHow the combined dataset splits across report types (`Data`, `Recovery`, `PendingAuthorizations`).")

code(r"""
report_counts = df_final["REPORT_TYPE"].value_counts()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(report_counts.index, report_counts.values, color=CATEGORICAL[: len(report_counts)], width=0.55)
ax.set_ylabel("Claims")
ax.set_title("Claims by report type", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="y")
for i, v in enumerate(report_counts.values):
    ax.text(i, v, f" {v:,}", ha="center", va="bottom", color=INK_SECONDARY, fontsize=9)
plt.tight_layout()
plt.show()
""")

md("### 8.2 Claim status distribution\n\nMost claims settle at `Settled` or `PO Issued` — this shows the full funnel across all statuses, most frequent first.")

code(r"""
status_counts = df_final["STATUS"].value_counts()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(status_counts.index[::-1], status_counts.values[::-1], color=CATEGORICAL[0], height=0.6)
ax.set_xlabel("Claims")
ax.set_title("Claim status distribution", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="x")
plt.tight_layout()
plt.show()
""")

md("### 8.3 Claims volume over time\n\nMonthly claim volume by transaction date (`TXNDATE`).")

code(r"""
monthly = (
    df_final.dropna(subset=["TXNDATE"])
    .set_index("TXNDATE")
    .resample("MS")
    .size()
)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(monthly.index, monthly.values, color=CATEGORICAL[0], linewidth=2)
ax.fill_between(monthly.index, monthly.values, color=CATEGORICAL[0], alpha=0.12)
ax.set_ylabel("Claims")
ax.set_title("Monthly claims volume (by transaction date)", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="y")
plt.tight_layout()
plt.show()
""")

md("### 8.4 Financial amounts: distribution\n\n`TOTALAMOUNTCLAIMED` is heavily right-skewed (a few very large claims), so this uses a log-scaled x-axis, capped at the 99th percentile to keep the shape readable.")

code(r"""
amt = df_final["TOTALAMOUNTCLAIMED"]
amt_positive = amt[amt > 0]
cap = amt_positive.quantile(0.99)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(amt_positive[amt_positive <= cap], bins=60, color=CATEGORICAL[2])
ax.set_xlabel("Total amount claimed")
ax.set_ylabel("Claims")
ax.set_title(f"Distribution of claimed amounts (capped at 99th pct = {cap:,.0f})", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="y")
fmt_thousands(ax, axis="x")
plt.tight_layout()
plt.show()

amt_positive.describe()
""")

md("### 8.5 Claimed vs. payable vs. payer share\n\nSummed across all claims — how much was claimed, how much was deemed payable, and how much the payer covered.")

code(r"""
totals = df_final[["TOTALAMOUNTCLAIMED", "TOTALPAYABLE", "PAYERSHARE"]].sum()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(totals.index, totals.values, color=CATEGORICAL[: len(totals)], width=0.5)
ax.set_ylabel("Total amount")
ax.set_title("Claimed vs. payable vs. payer share (sum across all claims)", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="y")
for i, v in enumerate(totals.values):
    ax.text(i, v, f" {v:,.0f}", ha="center", va="bottom", color=INK_SECONDARY, fontsize=9)
plt.tight_layout()
plt.show()
""")

md("### 8.6 Claims by provider type and emirate")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

pt = df_final["PROVIDERTYPE"].value_counts()
axes[0].bar(pt.index, pt.values, color=CATEGORICAL[3], width=0.6)
axes[0].set_title("Claims by provider type", color=INK, fontsize=11, loc="left")
axes[0].tick_params(axis="x", rotation=25)
fmt_thousands(axes[0], axis="y")

em = df_final["PROVIDER EMIRATE"].value_counts()
axes[1].barh(em.index[::-1], em.values[::-1], color=CATEGORICAL[4], height=0.6)
axes[1].set_title("Claims by provider emirate", color=INK, fontsize=11, loc="left")
fmt_thousands(axes[1], axis="x")

plt.tight_layout()
plt.show()
""")

md("### 8.7 Beneficiary age band and gender")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ageband_order = sorted(df_final["AGEBAND"].dropna().unique(), key=lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 999)
ab = df_final["AGEBAND"].value_counts().reindex(ageband_order)
axes[0].bar(ab.index, ab.values, color=CATEGORICAL[0], width=0.6)
axes[0].set_title("Claims by beneficiary age band", color=INK, fontsize=11, loc="left")
axes[0].tick_params(axis="x", rotation=45)
fmt_thousands(axes[0], axis="y")

gender = df_final["GENDER"].value_counts()
axes[1].pie(
    gender.values, labels=gender.index, autopct="%1.0f%%",
    colors=CATEGORICAL[:2], startangle=90,
    wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
    textprops={"color": INK},
)
axes[1].set_title("Claims by gender", color=INK, fontsize=11, loc="left")

plt.tight_layout()
plt.show()
""")

md("### 8.8 Post-termination flag\n\nClaims where the service date (`TXNDATE`) falls after the policy's `EXPIRYDATE` — a small number, but worth an explicit audit view.")

code(r"""
flag_counts = df_final["IS_POST_TERMINATION"].value_counts()
labels = ["Within policy period", "After policy expiry"]
values = [flag_counts.get(False, 0), flag_counts.get(True, 0)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, values, color=[CATEGORICAL[0], "#d03b3b"], width=0.5)
ax.set_ylabel("Claims")
ax.set_title("Post-termination audit flag", color=INK, fontsize=12, loc="left", pad=12)
fmt_thousands(ax, axis="y")
for i, v in enumerate(values):
    ax.text(i, v, f" {v:,}", ha="center", va="bottom", color=INK_SECONDARY, fontsize=9)
ax.set_yscale("symlog")
plt.tight_layout()
plt.show()
""")

md("### 8.9 Excluded claims (Rejected / Void)\n\nClaims that were pulled out of `df_final` because their status was `Rejected` or `Void`.")

code(r"""
if not df_excluded_claims.empty:
    exc_counts = df_excluded_claims["STATUS"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(exc_counts.index, exc_counts.values, color=[CATEGORICAL[7], CATEGORICAL[5]][: len(exc_counts)], width=0.5)
    ax.set_ylabel("Claims")
    ax.set_title("Excluded claims: Rejected vs. Void", color=INK, fontsize=12, loc="left", pad=12)
    fmt_thousands(ax, axis="y")
    for i, v in enumerate(exc_counts.values):
        ax.text(i, v, f" {v:,}", ha="center", va="bottom", color=INK_SECONDARY, fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("No excluded (Rejected/Void) claims in this dataset.")
""")

md("### 8.10 Recovery / ex-gratia amounts\n\nDistribution of recovery amounts among the claims that had a non-zero `RECOVERY/EX-GRATIA` value.")

code(r"""
if not df_recovery.empty:
    rec = df_recovery["RECOVERY/EX-GRATIA"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(rec, bins=40, color=CATEGORICAL[6])
    ax.set_xlabel("Recovery / ex-gratia amount")
    ax.set_ylabel("Claims")
    ax.set_title(f"Recovery/ex-gratia amounts (n={len(rec):,}, total={rec.sum():,.0f})", color=INK, fontsize=12, loc="left", pad=12)
    fmt_thousands(ax, axis="y")
    fmt_thousands(ax, axis="x")
    plt.tight_layout()
    plt.show()
else:
    print("No recovery/ex-gratia claims in this dataset.")
""")

md(r"""## Summary

| Dataset | Rows | Notes |
|---|---|---|
| `df_utilization` | combined raw claims | every sheet of every claims workbook |
| `df_valid_claims` | excludes Rejected/Void | |
| `df_final` | valid claims + beneficiary merge | exported as `consolidated_claims_all_reports.csv` |
| `df_excluded_claims` | Rejected/Void only | exported as `excluded_rejected_void_claims.csv` |
| `df_recovery` | non-zero recovery/ex-gratia | exported as `recovery_ex_gratia_claims.csv` |

All processed CSVs are written to `data/processed/`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

out_path = "notebooks/data_processing.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
