# PREG (Power Regulator) Alerts Explorer

Interactive Streamlit app to analyze **fan speed** behavior in power regulator modules.

## What it does
- **Factory defaults** rule: threshold `t=1026 RPM`, rolling window `w=2h`, hit limit `l=2`.
- **Recommended values** rule: per-PREG thresholds and limits (e.g., `PREG_FAN_2 = 1053 RPM`).
- **Distribution view**: histogram of `SampleValue` with a vertical threshold line to support threshold calibration.
- **Visuals**:
  - Bar chart: number of affected systems per `SampleID` (PREG).
  - Per-system timeline with a horizontal threshold line.
  - Distribution histogram with key summary stats (min, max, median).
- **Exports**: download filtered rows for each system as CSV.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements_preg.txt
```


## Run

streamlit run 2512_PREG.py

## Snapshots

.png files show the expected output of the code.
