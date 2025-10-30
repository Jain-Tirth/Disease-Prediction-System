Backend for Disease Prediction System

Purpose
- A small FastAPI app that exposes a /predict endpoint and /features endpoint.

Files to provide
- model.joblib : trained pipeline or estimator saved with joblib.dump(pipeline, 'backend/model.joblib')
- label_encoder.joblib (optional) : LabelEncoder saved with joblib.dump(le, 'backend/label_encoder.joblib')
- feature_names.json (optional but recommended) : JSON array of feature column names in order, e.g. ["fever","cough",...]

Quick save snippet (run in your notebook after training):

```python
import joblib
import json

# pipeline is your trained sklearn pipeline or estimator
joblib.dump(pipeline, 'backend/model.joblib')

# le is your LabelEncoder for the target
joblib.dump(le, 'backend/label_encoder.joblib')

# feature_cols is the ordered list of features used to train the model
with open('backend/feature_names.json', 'w', encoding='utf-8') as f:
    json.dump(feature_cols, f)
```

Run the server

```
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Notes
- The API accepts either an `input_vector` (list of 0/1 ints in feature order) or `symptoms` (list of symptom names present). If you supply symptoms, `feature_names.json` must be present.
