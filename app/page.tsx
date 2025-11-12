"use client";

import { useMemo, useState } from 'react';
import { trainLogisticRegression, predictProbability, type TrainedModel } from "@lib/logreg";
import { generateSyntheticDataset, featureVectorFromInputs, type InputFeatures } from "@lib/stemi";
import { parseCsvFeatures } from "@lib/csv";

export default function Page() {
  const [sex, setSex] = useState<'male' | 'female'>('male');
  const [age, setAge] = useState<number>(60);
  const [heartRate, setHeartRate] = useState<number>(80);
  const [elev, setElev] = useState<Record<string, number>>({
    I: 0, II: 0, III: 0, aVR: 0, aVL: 0, aVF: 0,
    V1: 0, V2: 0, V3: 0, V4: 0, V5: 0, V6: 0,
  });
  const [model, setModel] = useState<TrainedModel | null>(null);
  const [training, setTraining] = useState(false);
  const [prob, setProb] = useState<number | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);

  const exampleCsv = useMemo(() => (
    'sex,age,hr,I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n' +
    'male,63,90,0.5,1.2,1.8,-0.4,0.1,1.6,0.2,2.5,3.2,2.8,1.0,0.6,0.3\n'
  ), []);

  function setLead(name: string, value: number) {
    setElev(prev => ({ ...prev, [name]: value }));
  }

  async function handleTrain() {
    setTraining(true);
    setProb(null);
    try {
      const { X, y } = generateSyntheticDataset(4000, 42);
      const trained = trainLogisticRegression(X, y, { learningRate: 0.2, epochs: 1200, l2: 1e-4 });
      setModel(trained);
    } finally {
      setTraining(false);
    }
  }

  function handlePredict() {
    if (!model) return;
    const fv = featureVectorFromInputs({ sex, age, heartRate, elev });
    const p = predictProbability(model, fv);
    setProb(p);
  }

  async function onCsvUpload(file: File) {
    setCsvError(null);
    try {
      const text = await file.text();
      const rows = parseCsvFeatures(text);
      if (rows.length === 0) throw new Error('No rows found');
      const r = rows[0];
      setSex(r.sex);
      setAge(r.age);
      setHeartRate(r.heartRate);
      setElev(r.elev);
    } catch (e: any) {
      setCsvError(e?.message ?? 'Failed to parse CSV');
    }
  }

  const riskBadge = prob == null ? null : (
    <span className={`badge ${prob >= 0.5 ? 'warn' : 'ok'}`}>
      {prob >= 0.5 ? 'High STEMI likelihood' : 'Low STEMI likelihood'}
    </span>
  );

  return (
    <div className="container">
      <div className="card">
        <div className="h1">STEMI Detector</div>
        <p className="text-muted">Client-side demonstration using a simple logistic regression trained on synthetic data that mimics common ST-elevation rules. Not for clinical use.</p>

        <div className="row">
          <div className="col">
            <div className="h2">Patient</div>
            <label>Sex</label>
            <select value={sex} onChange={e => setSex(e.target.value as any)}>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
            <label>Age</label>
            <input type="number" value={age} onChange={e => setAge(parseInt(e.target.value || '0'))} />
            <label>Heart rate (bpm)</label>
            <input type="number" value={heartRate} onChange={e => setHeartRate(parseInt(e.target.value || '0'))} />

            <div className="h2">ST Elevation per lead (mm)</div>
            {Object.keys(elev).map((k) => (
              <div key={k} style={{ display: 'grid', gridTemplateColumns: '80px 1fr', alignItems: 'center', gap: 8 }}>
                <label>{k}</label>
                <input type="number" step="0.1" value={elev[k]} onChange={e => setLead(k, parseFloat(e.target.value || '0'))} />
              </div>
            ))}
          </div>

          <div className="col">
            <div className="h2">Model</div>
            <div className="text-muted small">Train a lightweight logistic regression on synthetic cases (runs in ~1-3s)</div>
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button onClick={handleTrain} disabled={training}>Train model</button>
              <button className="secondary" onClick={handlePredict} disabled={!model}>Predict</button>
            </div>
            {training && <p>Training...</p>}
            {model && (
              <div style={{ marginTop: 12 }}>
                <div className="small text-muted">Model trained: {model.weights.length} weights</div>
              </div>
            )}
            {prob != null && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 24, fontWeight: 700 }}>{(prob * 100).toFixed(1)}%</div>
                {riskBadge}
              </div>
            )}

            <div className="h2">CSV import</div>
            <div className="small text-muted">Header: sex,age,hr,I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6</div>
            <input type="file" accept="text/csv" onChange={e => e.target.files && onCsvUpload(e.target.files[0])} />
            {csvError && <div className="small" style={{ color: '#b91c1c' }}>{csvError}</div>}
            <details style={{ marginTop: 8 }}>
              <summary className="small">Example CSV</summary>
              <pre className="small" style={{ background: '#f1f5f9', padding: 8, borderRadius: 8 }}>{exampleCsv}</pre>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}
