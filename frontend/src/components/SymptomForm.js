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
  const [searchTerm, setSearchTerm] = useState('');

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

  function clearSelection() {
    const cleared = {};
    features.forEach((f) => (cleared[f] = false));
    setChecked(cleared);
    setResult(null);
    setError(null);
  }

  const selectedCount = features ? Object.values(checked).filter(Boolean).length : 0;

  const filteredFeatures = features
    ? features.filter((f) =>
        f.toLowerCase().replace(/_/g, ' ').includes(searchTerm.toLowerCase())
      )
    : [];

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let payload;
      if (features && Object.keys(checked).length > 0) {
        const symptoms = Object.keys(checked).filter((k) => checked[k]);
        if (symptoms.length === 0) {
          throw new Error('Please select at least one symptom');
        }
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
      <h2>Disease Prediction Tool</h2>
      <p className="symptom-form-intro">
        Select the symptoms you're experiencing from the list below. Our AI will analyze your symptoms and provide a disease prediction.
      </p>

      <form onSubmit={onSubmit}>
        {features ? (
          <>
            <div className="search-box">
              <span className="search-icon">🔎</span>
              <input
                type="text"
                className="search-input"
                placeholder="Search symptoms..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>

            {selectedCount > 0 && (
              <div className="selected-count">
                <span>{selectedCount}</span> symptom{selectedCount !== 1 ? 's' : ''} selected
              </div>
            )}

            <div className="features-grid">
              {filteredFeatures.map((f) => (
                <label
                  key={f}
                  className={`feature-item ${checked[f] ? 'selected' : ''}`}
                  onClick={() => toggle(f)}
                >
                  <input
                    type="checkbox"
                    checked={!!checked[f]}
                    onChange={() => {}}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <span>{f.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>

            {filteredFeatures.length === 0 && searchTerm && (
              <p style={{ textAlign: 'center', color: '#718096', padding: '2rem' }}>
                No symptoms found matching "{searchTerm}"
              </p>
            )}
          </>
        ) : (
          <div className="fallback-container">
            <p>
              ⚠️ Feature names are not available from the server. Please paste a comma-separated vector of
              0/1 values matching the model input order (132 values).
            </p>
            <textarea
              value={rawVector}
              onChange={(e) => setRawVector(e.target.value)}
              placeholder="0,1,0,0,1,0,1,..."
              rows={5}
            />
          </div>
        )}

        <div className="actions">
          <button type="submit" disabled={loading || (features && selectedCount === 0)}>
            {loading && <span className="loading-spinner"></span>}
            {loading ? 'Analyzing...' : 'Get Prediction'}
          </button>
          {features && selectedCount > 0 && (
            <button type="button" onClick={clearSelection}>
              Clear Selection
            </button>
          )}
        </div>
      </form>

      {error && (
        <div className="error">
          <strong>❌ Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result">
          <h3>✅ Prediction Result</h3>
          <p>
            <strong>Disease:</strong> {result.prediction}
          </p>
          <p>
            <strong>Confidence Code:</strong> {result.raw_prediction}
          </p>
          <p style={{ fontSize: '0.9rem', marginTop: '1rem', opacity: 0.8 }}>
            ℹ️ This is an AI prediction based on your symptoms. Please consult a healthcare professional for proper medical diagnosis and treatment.
          </p>
        </div>
      )}
    </div>
  );
}
