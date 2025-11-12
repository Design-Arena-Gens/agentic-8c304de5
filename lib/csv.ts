import type { InputFeatures } from "@lib/stemi";

export type CsvRow = InputFeatures;

export function parseCsvFeatures(csvText: string): CsvRow[] {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const header = lines[0].split(',').map(s => s.trim());
  const idx = Object.fromEntries(header.map((h, i) => [h, i] as const));
  const required = ['sex','age','hr','I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'];
  for (const r of required) if (!(r in idx)) throw new Error(`Missing column: ${r}`);

  const rows: CsvRow[] = [];
  for (let li = 1; li < lines.length; li++) {
    const parts = lines[li].split(',');
    if (parts.length < header.length) continue;
    const sexRaw = parts[idx['sex']].trim().toLowerCase();
    const sex = sexRaw === 'male' ? 'male' : sexRaw === 'female' ? 'female' : (()=>{ throw new Error(`Invalid sex '${sexRaw}'`); })();
    const age = parseFloat(parts[idx['age']]);
    const hr = parseFloat(parts[idx['hr']]);
    const elev = {
      I: parseFloat(parts[idx['I']]),
      II: parseFloat(parts[idx['II']]),
      III: parseFloat(parts[idx['III']]),
      aVR: parseFloat(parts[idx['aVR']]),
      aVL: parseFloat(parts[idx['aVL']]),
      aVF: parseFloat(parts[idx['aVF']]),
      V1: parseFloat(parts[idx['V1']]),
      V2: parseFloat(parts[idx['V2']]),
      V3: parseFloat(parts[idx['V3']]),
      V4: parseFloat(parts[idx['V4']]),
      V5: parseFloat(parts[idx['V5']]),
      V6: parseFloat(parts[idx['V6']]),
    } as const;

    rows.push({ sex, age, heartRate: hr, elev });
  }
  return rows;
}
