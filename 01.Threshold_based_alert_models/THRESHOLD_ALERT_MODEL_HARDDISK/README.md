# Hard Disk Alerts Explorer

Interactive Streamlit app to identify systems reporting **low disk free space** and to review per-system timelines.

## What it does
- **Factory defaults** rule: flags systems when `% free space ≤ 5` with **≥ 2 hits within any rolling 48-hour window**.
- **Recommended values** rule: per-partition thresholds, rolling-window hours, and hit limits (e.g., `E:` uses a higher threshold with a shorter window).
- **Visuals**:
  - Bar chart: number of affected systems per partition (`SamplingID`).
  - Per-system timeline with a horizontal threshold line.
- **Exports**: download filtered rows for each system as CSV.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements_harddisk.txt

## How to run
streamlit run 2511_AlertExplore_HardDisk.py

