Full-stack setup (React frontend + FastAPI backend)

What I added
- `backend/` - FastAPI app (`app.py`), `requirements.txt` and README explaining how to save model and run server.
- `frontend/src/components/SymptomForm.js` - React UI to select symptoms or enter a raw vector and call the backend.
- `frontend/src/api.js` - small helper to talk to the backend.
- Updated `frontend/src/App.js` to mount the component.

Quick start (development)

1) Prepare and run the backend

 - From the repo root (or the backend folder):

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
# Make sure backend/model.joblib and backend/feature_names.json (optional) are present
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

2) Run the React frontend

 - From `frontend` folder:

```
npm install
npm start
```

Configuration

- By default the React app calls `http://localhost:8000`. To change, set `REACT_APP_API_BASE` in `.env` inside `frontend`.

Notes

- You must export your trained sklearn pipeline and optional LabelEncoder and feature_names.json to the `backend/` folder as explained in `backend/README.md`.
- The frontend will try to fetch the feature names from `/features`. If they are available, it renders checkboxes. Otherwise it asks for a raw vector.
