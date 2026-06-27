============================================================
  VERITAS v2.0 — HOW TO RUN
============================================================

FOLDER STRUCTURE (must look exactly like this):
------------------------------------------------------------
veritas_final/
├── app.py
├── requirements.txt
├── HOW_TO_RUN.txt
└── frontend/
    └── index.html

------------------------------------------------------------
STEP 1 — Open terminal in the veritas_final folder
------------------------------------------------------------

  Windows PowerShell:
    cd C:\Users\Prajwal\OneDrive\Desktop\veritas_final

  Or right-click the veritas_final folder → "Open in Terminal"

------------------------------------------------------------
STEP 2 — Create virtual environment (first time only)
------------------------------------------------------------

  python -m venv .venv

------------------------------------------------------------
STEP 3 — Activate virtual environment
------------------------------------------------------------

  Windows:
    .venv\Scripts\activate

  Mac/Linux:
    source .venv/bin/activate

  You will see (.venv) at the start of your terminal line.

------------------------------------------------------------
STEP 4 — Install dependencies (first time only)
------------------------------------------------------------

  pip install -r requirements.txt

------------------------------------------------------------
STEP 5 — Run the app
------------------------------------------------------------

  python app.py

  You should see:
    * Running on http://127.0.0.1:5000
    * Running on http://10.x.x.x:5000

------------------------------------------------------------
STEP 6 — Open in browser
------------------------------------------------------------

  Go to:  http://127.0.0.1:5000

  Use API key:  demo-key-2024

------------------------------------------------------------
STOP the server:  Press Ctrl+C in terminal
============================================================
