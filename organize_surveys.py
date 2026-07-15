"""
================================================================================
Phase 1: SMART Surveys Organization & Categorization
================================================================================
Data Science Final Project Pipeline - Phase 1

Objective:
    - Ingest all raw .as survey data
    - Eliminate any duplicate files based on content and filename
    - Categorize surveys into Individual or Aggregate formats
    - Group surveys by Admin2 or LHZ boundaries
    - Apply a standardized naming convention to all files
    - Route the processed files into appropriate subdirectories
    - Parse out and format mortality data tables into clean CSV files

Inputs:  Raw '.as' text files located in the 'Smart Survey Data Sets' directory
Outputs: Structured CSV files distributed into:
         output/admin2_surveys/individual_surveys/
         output/admin2_surveys/aggregate_surveys/
         output/admin2_surveys/issue_surveys/
         output/lhz_surveys/individual_surveys/
================================================================================
"""

# ==============================================================================
# 1) Imports
# ==============================================================================

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import time
from io import StringIO
from pathlib import Path

# ==============================================================================
# 2) Configuration — Change these paths for Google Colab or another machine
# ==============================================================================

# For LOCAL use: resolves to the folder containing this script
WORKSPACE_ROOT = Path(__file__).parent.resolve()

# For GOOGLE COLAB: uncomment the two lines below and comment out the line above
# from google.colab import drive
# drive.mount('/content/drive')
# WORKSPACE_ROOT = Path("/content/drive/MyDrive/A A Data")

# Source folder containing the raw .as files
DATA_ROOT = WORKSPACE_ROOT / "Smart Survey Data Sets" / "Smart Survey Data Sets"

# Output directories
RAW_ROOT     = WORKSPACE_ROOT / "raw data"
RAW_CSV_ROOT = RAW_ROOT / "csv"
OUTPUT_ROOT  = WORKSPACE_ROOT / "output"

ADMIN_IND   = OUTPUT_ROOT / "admin2_surveys" / "individual_surveys"
ADMIN_AGG   = OUTPUT_ROOT / "admin2_surveys" / "aggregate_surveys"
ADMIN_ISSUE = OUTPUT_ROOT / "admin2_surveys" / "issue_surveys"
LHZ_IND     = OUTPUT_ROOT / "lhz_surveys"    / "individual_surveys"

ALL_OUTPUT_DIRS = (RAW_ROOT, RAW_CSV_ROOT, ADMIN_IND, ADMIN_AGG, ADMIN_ISSUE, LHZ_IND)

# ==============================================================================
# 3) Helper Functions — Directory Management
# ==============================================================================


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for p in ALL_OUTPUT_DIRS:
        p.mkdir(parents=True, exist_ok=True)


def clear_generated_outputs() -> None:
    """Delete all previously generated files so we start fresh each run."""
    for folder in ALL_OUTPUT_DIRS:
        for f in folder.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass


# ==============================================================================
# 4) Helper Functions — File Naming & Deduplication
# ==============================================================================


def normalized_name(name: str) -> str:
    """
    Normalize a filename for deduplication.
    Strips ' - Copy' suffixes, lowercases, and replaces non-alphanumeric
    characters with underscores.
    """
    stem = Path(name).stem.lower()
    stem = re.sub(r"\s*-\s*copy$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def infer_place(file_name: str, year: str, month: str) -> str:
    """
    Extract the place/district name from a filename by removing known
    stopwords (som, smart, survey, admin2, lhz, etc.) and date tokens.
    Returns the cleaned place name in uppercase (e.g. 'BAYDHABA').
    """
    stem = Path(file_name).stem.lower()
    stem = re.sub(r"\s*-\s*copy$", "", stem)
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]

    stopwords = {
        "som", "smart", "survey", "surveys", "admin1", "admin2", "lhz",
        "ena", "anthro", "mortality", "complete", "final", "copy",
        "idp", "urban", "rural", "pastoral", "agropast", "agropastoral",
        "pump", "riverine", "revine",
    }

    place_tokens: list[str] = []
    for t in tokens:
        if t in stopwords:
            continue
        if re.fullmatch(r"20\d{2}", t):        # skip 4-digit year
            continue
        if re.fullmatch(r"\d{6}", t):           # skip YYYYMM patterns
            continue
        if month and re.fullmatch(rf"{int(month):02d}", t):
            continue
        if year and t == year:
            continue
        place_tokens.append(t)

    if not place_tokens:
        return "UNKNOWN"
    return "_".join(tok.upper() for tok in place_tokens)


def standardized_name(file_name: str, year: str, month: str,
                      level: str, fmt: str) -> str:
    """
    Build a clean standardized output filename.
    Format: SOM_YYYY_MM_PLACE_level_format.csv
    Example: SOM_2018_07_ADDUN_lhz_individual.csv
    """
    mm   = f"{int(month):02d}" if month.isdigit() else "00"
    yyyy = year if year.isdigit() and len(year) == 4 else "0000"
    place = infer_place(file_name, yyyy, mm)
    return f"SOM_{yyyy}_{mm}_{place}_{level}_{fmt}.csv"


# ==============================================================================
# 5) Helper Functions — Date Parsing
# ==============================================================================


def get_date_order(lines: list[str]) -> str:
    """Detect date format order (mdy, dmy, ymd) from the .as file header."""
    for ln in lines:
        t = ln.strip().lower()
        if t in {"mdy", "dmy", "ymd"}:
            return t
    return "mdy"


def parse_month_year_from_date(date_text: str, order: str) -> tuple[str, str]:
    """Parse a single date string (e.g. '7/12/2018') into (month, year)."""
    m = re.search(r"(?P<a>\d{1,4})[/-](?P<b>\d{1,2})[/-](?P<c>\d{1,4})", date_text)
    if not m:
        return "", ""

    a, b, c = int(m.group("a")), int(m.group("b")), int(m.group("c"))

    if order == "ymd":
        year_val  = a if a >= 100 else 2000 + a
        month_val = b
    elif order == "dmy":
        year_val  = c if c >= 100 else 2000 + c
        month_val = b
    else:  # mdy (default)
        year_val  = c if c >= 100 else 2000 + c
        month_val = a

    if not (1 <= month_val <= 12):
        return "", ""
    return str(month_val), str(year_val)


def parse_month_year(lines: list[str], file_name: str) -> tuple[str, str]:
    """
    Try to determine the survey month and year.
    Strategy: (1) parse from the first date in the file,
              (2) fall back to the filename pattern.
    """
    order = get_date_order(lines)

    # Strategy 1: first date found in the file content
    text = "\n".join(lines)
    m = re.search(r"(?P<a>\d{1,4})[/-](?P<b>\d{1,2})[/-](?P<c>\d{1,4})", text)
    if m:
        a, b, c = int(m.group("a")), int(m.group("b")), int(m.group("c"))
        year = c if c >= 100 else 2000 + c
        if order == "mdy":
            return str(a), str(year)
        if order == "dmy":
            return str(b), str(year)
        if order == "ymd":
            y = a if a >= 100 else 2000 + a
            return str(b), str(y)

    # Strategy 2: extract YYYY_MM or YYYYMM from filename
    nm = re.search(r"(20\d{2})[_\-]?(0[1-9]|1[0-2])", file_name)
    if nm:
        return str(int(nm.group(2))), nm.group(1)

    # Strategy 3: extract just the year from filename
    y_only = re.search(r"(20\d{2})", file_name)
    if y_only:
        return "", y_only.group(1)

    return "", ""


def get_month_year_from_survdate_column(lines: list[str],
                                        order: str) -> tuple[str, str]:
    """
    Scan the anthropometry table for a SURVDATE column and extract
    the month/year from the first data row.
    """
    in_surv_table = False
    survdate_idx  = -1

    for line in lines:
        lower = line.lower()
        cells = split_cells(line)
        lowered_cells = [c.strip().lower() for c in cells]

        # Stop before the mortality blocks
        if "mortality_new" in lower or "mor_individual" in lower \
                or "mortality_individual" in lower:
            break

        if not in_surv_table and "survdate" in lowered_cells:
            in_surv_table = True
            survdate_idx  = lowered_cells.index("survdate")
            continue

        if not in_surv_table:
            continue
        if not line.strip():
            continue
        if survdate_idx < 0 or survdate_idx >= len(cells):
            continue

        date_text = cells[survdate_idx].strip()
        if not date_text:
            continue

        month, year = parse_month_year_from_date(date_text, order)
        if month and year:
            return month, year

    return "", ""


# ==============================================================================
# 6) Helper Functions — Survey Classification
# ==============================================================================


def split_cells(line: str) -> list[str]:
    """Split a tab-delimited line into cells."""
    return line.split("\t")


def has_aggregate_data(lines: list[str]) -> bool:
    """
    Check if the file contains aggregate mortality data.
    Aggregate data lives under the '?Mortality_new:' block and has
    rows where the first cell is a number (household ID).
    """
    in_agg = False
    for line in lines:
        lower = line.lower()
        if "mortality_new" in lower:
            in_agg = True
            continue
        if in_agg and not line.strip():
            in_agg = False
            continue
        if in_agg and line.strip():
            first = split_cells(line)[0].strip() if split_cells(line) else ""
            if re.fullmatch(r"\d+", first):
                return True
    return False


def has_individual_data(lines: list[str]) -> bool:
    """
    Check if the file contains individual mortality data.
    Individual data lives under '?Mor_individual:' and has a header row
    with columns like P1_sex, P1_age, and actual data rows below it.
    """
    in_ind = False
    header_found = False
    for line in lines:
        lower = line.lower()
        if "mor_individual" in lower or "mortality_individual" in lower:
            in_ind = True
            header_found = False
            continue
        if in_ind and not line.strip():
            in_ind = False
            continue
        if in_ind and line.strip():
            cells = split_cells(line)
            t = line.lower()
            if (not header_found) and ("p1_sex" in t and "p1_age" in t):
                header_found = True
                continue
            if header_found and len(cells) > 3:
                if any(c.strip() for c in cells[1:]):
                    return True
    return False


# ==============================================================================
# 7) Helper Functions — CSV Writing
# ==============================================================================


def row_to_csv(cells: list[str]) -> str:
    """Convert a list of cell values into a single CSV-formatted line."""
    s = StringIO()
    writer = csv.writer(s, lineterminator="")
    writer.writerow(cells)
    return s.getvalue()


def write_raw_csv(src_text: str, out_csv: Path) -> None:
    """Write the raw .as content to CSV by replacing tabs with commas."""
    lines = src_text.splitlines()
    out_csv.write_text(
        "\n".join(ln.replace("\t", ",") for ln in lines),
        encoding="utf-8"
    )


def safe_write(path: Path, lines: list[str]) -> bool:
    """Write lines to file with retry logic for OneDrive/cloud sync issues."""
    payload = "\n".join(lines)
    for i in range(3):
        try:
            path.write_text(payload, encoding="utf-8")
            return True
        except Exception as e:
            if i == 2:
                print(f"DEBUG: safe_write failed for {path}. Error: {e}")
            time.sleep(0.4)
    return False


# ==============================================================================
# 8) Core Function — Extract & Build Clean Mortality CSV Lines
# ==============================================================================


def extract_month_year_from_output_lines(lines: list[str]) -> tuple[str, str]:
    """
    After building output lines, check if the first data row contains
    valid month/year values. Used to refine the date for filename generation.
    """
    for idx, line in enumerate(lines):
        cells = [c.strip().lower() for c in split_cells(line)]
        if len(cells) >= 2 and cells[0] == "month" and cells[1] == "year":
            for data_line in lines[idx + 1:]:
                data_cells = [c.strip() for c in split_cells(data_line)]
                if len(data_cells) < 2:
                    continue
                month, year = data_cells[0], data_cells[1]
                if re.fullmatch(r"\d{1,2}", month) and re.fullmatch(r"\d{4}", year):
                    if 1 <= int(month) <= 12:
                        return month, year
    return "", ""


def extract_recall_period(lines: list[str]) -> int:
    """
    Search the ?Planning: block for the parameter row and extract the recall period.
    Returns the recall period as an integer, defaulting to 90 if not found.
    """
    in_plan = False
    for line in lines:
        lower = line.lower()
        if "planning:" in lower:
            in_plan = True
            continue
        if "training_new:" in lower or "mortality_new:" in lower or "mor_individual:" in lower:
            in_plan = False
        if in_plan:
            m = re.match(r"^\s*(\d+)\s*\t\s*([\d\.]+)\s*\t\s*(\d+)\s*$", line)
            if m:
                return int(m.group(1))
    return 90


def build_output_lines(lines: list[str], default_month: str,
                       default_year: str, recall_period: int) -> list[str]:
    """
    Walk through the .as file lines and extract ONLY the mortality table.
    Prepends month, year, and recall_period columns to each row.
    """
    out: list[str] = []
    date_order = get_date_order(lines)

    # Try to get a more accurate month/year from the SURVDATE column
    surv_month, surv_year = get_month_year_from_survdate_column(lines, date_order)
    if surv_month and surv_year:
        default_month, default_year = surv_month, surv_year

    has_agg = has_aggregate_data(lines)
    has_ind = has_individual_data(lines)

    if has_agg:
        # Standard columns for aggregate surveys
        headers = ["month", "year", "recall_period", "HH", "Cluster", "Team", "Total", "Births", "Deaths", "Joined", "Left", "Total_U5", "Births_U5", "Deaths_U5"]
        out.append(row_to_csv(headers))

        in_agg = False
        for line in lines:
            lower = line.lower()
            if "mortality_new" in lower:
                in_agg = True
                continue
            if in_agg and not line.strip():
                in_agg = False
                continue
            if in_agg:
                cells = split_cells(line)
                first = cells[0].strip() if cells else ""
                if re.fullmatch(r"\d+", first):
                    clean_cells = [c.strip() for c in cells]
                    clean_cells = clean_cells[:11]
                    if len(clean_cells) < 11:
                        clean_cells += ["0"] * (11 - len(clean_cells))
                    out.append(row_to_csv([default_month, default_year, str(recall_period)] + clean_cells))

    elif has_ind:
        in_ind = False
        header_found = False
        surdate_idx = -1
        date_idx = -1

        for line in lines:
            lower = line.lower()
            if "mor_individual" in lower or "mortality_individual" in lower:
                in_ind = True
                header_found = False
                continue
            if in_ind and not line.strip():
                in_ind = False
                continue
            if in_ind:
                cells = split_cells(line)
                lowered_cells = [c.strip().lower() for c in cells]

                if not header_found and ("surdate" in lowered_cells or "date" in lowered_cells):
                    header_found = True
                    if "surdate" in lowered_cells:
                        surdate_idx = lowered_cells.index("surdate")
                    else:
                        surdate_idx = lowered_cells.index("date")
                    date_idx = lowered_cells.index("date") if "date" in lowered_cells else -1

                    header_cells = [c.strip() for c in cells]
                    if date_idx >= 0 and date_idx < len(header_cells):
                        del header_cells[date_idx]
                    header_cells = ["month", "year", "recall_period"] + header_cells
                    out.append(row_to_csv(header_cells))
                    continue

                if header_found and len(cells) > 3:
                    if not any(c.strip() for c in cells[1:]):
                        continue
                    row_cells = [c.strip() for c in cells]
                    surdate_value = row_cells[surdate_idx] if 0 <= surdate_idx < len(row_cells) else ""
                    row_month, row_year = parse_month_year_from_date(surdate_value, date_order)
                    if not row_month or not row_year:
                        row_month, row_year = default_month, default_year

                    if date_idx >= 0 and date_idx < len(row_cells):
                        del row_cells[date_idx]

                    row_cells = [row_month, row_year, str(recall_period)] + row_cells
                    out.append(row_to_csv(row_cells))

    return out


# ==============================================================================
# 9) Main Pipeline — Stage 1: Organize and Classify
# ==============================================================================


def main() -> None:
    """
    Main execution loop for Phase 1.
    Processes all survey data, categorizes them, and extracts mortality tables.
    """
    print("=" * 60)
    print("Phase 1: SMART Surveys Organization & Categorization")
    print("=" * 60)

    # 1. Initialize output directories and clear previous runs
    ensure_dirs()
    clear_generated_outputs()

    # Step 2: Load all .as files
    files = sorted(DATA_ROOT.glob("*.as"))
    print(f"\nFound {len(files)} total .as files in:\n  {DATA_ROOT}\n")

    # Deduplication trackers
    seen_name: set[str] = set()
    seen_hash: set[str] = set()
    failed_writes: list[str] = []

    # Counters
    processed = duplicates = ind_count = agg_count = issue_count = 0
    admin2_ind = lhz_ind = 0
    used_output_names: dict[Path, int] = {}

    # Step 3: Process each file
    for file in files:
        # --- Deduplication by normalized name and content hash ---
        norm = normalized_name(file.name)
        content = file.read_text(encoding="utf-8", errors="replace")
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if norm in seen_name or sha in seen_hash:
            duplicates += 1
            continue

        seen_name.add(norm)
        seen_hash.add(sha)
        processed += 1

        # --- Copy raw file to raw data folder ---
        shutil.copy2(file, RAW_ROOT / file.name)
        write_raw_csv(content, RAW_CSV_ROOT / f"{file.stem}.csv")

        # --- Parse metadata ---
        lines = content.splitlines()
        month, year = parse_month_year(lines, file.name)
        has_agg = has_aggregate_data(lines)
        has_ind = has_individual_data(lines)
        is_lhz  = "lhz" in file.name.lower()

        # --- Classify the survey ---
        if has_agg:
            target = ADMIN_AGG
            level  = "admin2"
            fmt    = "aggregate"
            agg_count += 1
        elif has_ind:
            target = LHZ_IND if is_lhz else ADMIN_IND
            level  = "lhz"    if is_lhz else "admin2"
            fmt    = "individual"
            ind_count += 1
            if is_lhz:
                lhz_ind += 1
            else:
                admin2_ind += 1
        else:
            target = ADMIN_ISSUE
            level  = "admin2"
            fmt    = "issue"
            issue_count += 1

        # --- Extract clean mortality CSV lines ---
        recall_period = extract_recall_period(lines)
        out_lines = build_output_lines(lines, month, year, recall_period)
        out_month, out_year = extract_month_year_from_output_lines(out_lines)
        if out_month and out_year:
            month, year = out_month, out_year

        # --- Generate standardized filename ---
        base_name = standardized_name(file.name, year, month, level, fmt)
        out_path  = target / base_name

        # Handle duplicate output names by appending a suffix
        while out_path in used_output_names:
            used_output_names[out_path] += 1
            suffix   = used_output_names[out_path]
            out_path = target / base_name.replace(".csv", f"_{suffix}.csv")
        used_output_names[out_path] = 1

        # --- Write the clean CSV ---
        ok = safe_write(out_path, out_lines)
        if not ok:
            failed_writes.append(str(out_path))

    # 4. Display the execution summary
    print("-" * 60)
    print("Execution Summary")
    print("-" * 60)
    print(f"  Total distinct files handled : {processed}")
    print(f"  Duplicate files ignored      : {duplicates}")
    print(f"  Individual-level surveys     : {ind_count}")
    print(f"    -> Admin2 individual       : {admin2_ind}")
    print(f"    -> LHZ individual          : {lhz_ind}")
    print(f"  Aggregate-level surveys      : {agg_count}")
    print(f"  Flagged issue surveys        : {issue_count}")
    print("-" * 60)

    if failed_writes:
        print(f"\nWARNING: Failed to write {len(failed_writes)} file(s):")
        for fw in failed_writes:
            print(f"  - {fw}")

    print("\n[DONE] Stage 1 complete. Output saved to:")
    print(f"  {OUTPUT_ROOT}")


# ==============================================================================
# 10) Entry Point
# ==============================================================================

if __name__ == "__main__":
    main()
