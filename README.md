## How to Run

Each project keeps its own `requirements.txt` or `pyproject.toml`.

```bash
# Example: run ACB anomaly app
cd acb_temp_anomalies
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
``
