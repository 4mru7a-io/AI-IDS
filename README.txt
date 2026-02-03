AI-Based Network Intrusion Detection System (IDS)

Files:
- ai_ids.py       : Training & utility script (train & save model as model.pkl)
- app.py          : Flask dashboard (loads model.pkl if present)
- templates/       : HTML template for dashboard
- requirements.txt : Python dependencies
- report.docx     : Project report (Word document)

How to use:
1. Create a virtual environment: python -m venv venv
2. Activate it and install requirements: pip install -r requirements.txt
3. Place NSL-KDD CSV as nsl_kdd.csv in the project folder to train the model.
   Then run: python ai_ids.py --train
4. Start dashboard: python app.py
5. Open http://127.0.0.1:5000 to view the dashboard.

Notes:
- The included code is ready-to-run but does not include the dataset.
- For real deployments, secure the Flask API and persist alerts to a database.
