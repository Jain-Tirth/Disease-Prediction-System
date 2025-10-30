import React, { useEffect, useState } from 'react';
import { getFeatures, predict } from '../api';
import './SymptomForm.css';

export default function SymptomForm() {
  const [features, setFeatures] = useState(null);
  const [checked, setChecked] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [rawVector, setRawVector] = useState('');

  useEffect(() => {
    let mounted = true;
    getFeatures()
      .then((data) => {
        if (!mounted) return;
        setFeatures(data.features);
        const initial = {};
        data.features.forEach((f) => (initial[f] = false));
        setChecked(initial);
      })
      .catch(() => {
        // features not available — user will provide raw vector
        setFeatures(null);
      });
    return () => (mounted = false);
  }, []);

  function toggle(name) {
    setChecked((c) => ({ ...c, [name]: !c[name] }));
  }

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let payload;
      if (features && Object.keys(checked).length > 0) {
        const symptoms = Object.keys(checked).filter((k) => checked[k]);
        payload = { symptoms };
      } else {
        // parse raw vector
        const parts = rawVector.split(',').map((s) => s.trim()).filter(Boolean);
        const vec = parts.map((p) => (p === '1' || p.toLowerCase() === 'true' ? 1 : 0));
        payload = { input_vector: vec };
      }

      const res = await predict(payload);
      setResult(res);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="symptom-form">
      <form onSubmit={onSubmit}>
        {features ? (
          <div className="features-grid">
            {features.map((f) => (
              <label key={f} className="feature-item">
                <input type="checkbox" checked={!!checked[f]} onChange={() => toggle(f)} />
                {f}
              </label>
            ))}
          </div>
        ) : (
          <div>
            <p>
              Feature names are not available on the server. Paste a comma-separated vector of
              0/1 values matching the model input order (e.g. 0,1,0,0,1)
            </p>
            <textarea
              value={rawVector}
              onChange={(e) => setRawVector(e.target.value)}
              placeholder="0,1,0,0,1"
              rows={3}
            />
          </div>
        )}

        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Disease'}
          </button>
        </div>
      </form>

      {error && <div className="error">Error: {error}</div>}
      {result && (
        <div className="result">
          <h3>Prediction</h3>
          <p>
            <strong>Label:</strong> {result.prediction}
          </p>
          <p>
            <strong>Raw:</strong> {result.raw_prediction}
          </p>
        </div>
      )}
    </div>
  );
}
