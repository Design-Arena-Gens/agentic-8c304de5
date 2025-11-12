export type InputFeatures = {
  sex: 'male' | 'female';
  age: number;
  heartRate: number;
  elev: Record<string, number>; // ST elevation per lead in mm
};

const LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'] as const;

type Lead = typeof LEADS[number];

export function featureVectorFromInputs(inp: InputFeatures): number[] {
  const sexNum = inp.sex === 'male' ? 1 : 0;
  const ageScaled = (inp.age - 50) / 20; // center ~50y
  const hrScaled = (inp.heartRate - 70) / 30; // center ~70 bpm

  const stVals = LEADS.map(l => (inp.elev[l] ?? 0));

  // Region aggregates to capture contiguity-like signals
  const anterior = (inp.elev['V2'] ?? 0) + (inp.elev['V3'] ?? 0) + (inp.elev['V4'] ?? 0);
  const inferior = (inp.elev['II'] ?? 0) + (inp.elev['III'] ?? 0) + (inp.elev['aVF'] ?? 0);
  const lateral = (inp.elev['I'] ?? 0) + (inp.elev['aVL'] ?? 0) + (inp.elev['V5'] ?? 0) + (inp.elev['V6'] ?? 0);

  const maxLead = Math.max(...stVals);
  const sumAll = stVals.reduce((a,b)=>a+b,0);

  // Final feature vector: demographics + per-lead + aggregates
  return [sexNum, ageScaled, hrScaled, ...stVals, anterior, inferior, lateral, maxLead, sumAll];
}

function rand(seed: number) {
  // simple LCG
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function sampleNormal(r: () => number, mean = 0, std = 1) {
  // Box-Muller
  const u1 = Math.max(r(), 1e-9);
  const u2 = Math.max(r(), 1e-9);
  const mag = Math.sqrt(-2.0 * Math.log(u1));
  const z0 = mag * Math.cos(2 * Math.PI * u2);
  return mean + std * z0;
}

export function generateSyntheticDataset(n: number, seed = 123): { X: number[][]; y: number[] } {
  const r = rand(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const sex: 'male' | 'female' = r() < 0.5 ? 'male' : 'female';
    const age = Math.round(20 + r() * 70);
    const heartRate = Math.round(55 + r() * 80);
    const elev: Record<Lead, number> = {
      I: 0, II: 0, III: 0, aVR: 0, aVL: 0, aVF: 0, V1: 0, V2: 0, V3: 0, V4: 0, V5: 0, V6: 0,
    };

    // Background noise
    for (const l of LEADS) {
      elev[l] = Math.max(0, sampleNormal(r, 0.2, 0.25));
    }

    // Inject STEMI-like patterns in a subset
    const isStemi = r() < 0.35; // ~35% prevalence in synthetic data
    if (isStemi) {
      const territory = r();
      if (territory < 0.5) {
        // Anterior STEMI: V2, V3, V4 often highest
        const base = sex === 'male' ? 1.8 : 1.5;
        elev['V2'] += base + r() * 2.2; // up to ~4mm
        elev['V3'] += base + r() * 2.5;
        elev['V4'] += 0.8 + r() * 1.6;
      } else if (territory < 0.8) {
        // Inferior: II, III, aVF
        elev['II'] += 1.2 + r() * 2.0;
        elev['III'] += 1.5 + r() * 2.2;
        elev['aVF'] += 1.0 + r() * 1.8;
        // reciprocal depression in aVL sometimes
        elev['aVL'] = Math.max(0, elev['aVL'] - (0.3 + r() * 0.6));
      } else {
        // High lateral: I, aVL, V5, V6
        elev['I'] += 0.8 + r() * 1.6;
        elev['aVL'] += 1.0 + r() * 1.8;
        elev['V5'] += 0.9 + r() * 1.6;
        elev['V6'] += 0.8 + r() * 1.5;
      }
    }

    // Build features and label using rule-ish heuristic
    const fv = featureVectorFromInputs({ sex, age, heartRate, elev });

    const v2 = elev['V2'];
    const v3 = elev['V3'];
    const male = sex === 'male';
    const anteriorRule = (v2 + v3) >= (male ? 4.0 : 3.5);
    const inferiorRule = (elev['II'] + elev['III'] + elev['aVF']) >= 4.0 && Math.max(elev['II'], elev['III']) >= 1.0;
    const lateralRule = (elev['I'] + elev['aVL'] + elev['V5'] + elev['V6']) >= 3.0 && (elev['aVL'] >= 1.0 || elev['V5'] >= 1.0 || elev['V6'] >= 1.0);

    const label = (anteriorRule || inferiorRule || lateralRule) ? 1 : 0;

    X.push(fv);
    y.push(label);
  }
  return { X, y };
}
