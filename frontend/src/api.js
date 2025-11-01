const BASE = process.env.REACT_APP_API_BASE ;

export async function getFeatures() {
  console.log(BASE);
  const res = await fetch(`${BASE}features`);
  if (!res.ok) throw new Error('Failed to load features');
  return res.json();
}

export async function predict(payload) {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Prediction request failed');
  }
  return res.json();
}
