export type TrainedModel = {
  weights: number[]; // length = nFeatures + 1 (bias at end)
};

export type TrainOptions = {
  learningRate?: number;
  epochs?: number;
  l2?: number; // ridge regularization strength
};

export function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function addBias(x: number[]): number[] {
  return [...x, 1];
}

export function predictProbability(model: TrainedModel, x: number[]): number {
  const xb = addBias(x);
  return sigmoid(dot(model.weights, xb));
}

export function trainLogisticRegression(
  X: number[][],
  y: number[],
  options: TrainOptions = {}
): TrainedModel {
  if (X.length === 0) throw new Error('Empty dataset');
  const n = X.length;
  const d = X[0].length + 1; // +bias
  const lr = options.learningRate ?? 0.1;
  const epochs = options.epochs ?? 1000;
  const l2 = options.l2 ?? 0;

  let w = new Array(d).fill(0).map(() => (Math.random() - 0.5) * 0.01);

  const xWithBias = X.map(row => addBias(row));

  for (let epoch = 0; epoch < epochs; epoch++) {
    // compute gradient
    const grad = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const p = sigmoid(dot(w, xWithBias[i]));
      const err = p - y[i];
      for (let j = 0; j < d; j++) {
        grad[j] += err * xWithBias[i][j];
      }
    }
    // average and add L2 reg (excluding bias)
    for (let j = 0; j < d; j++) {
      grad[j] /= n;
      if (j !== d - 1 && l2 > 0) grad[j] += l2 * w[j];
    }
    // update
    for (let j = 0; j < d; j++) {
      w[j] -= lr * grad[j];
    }
  }

  return { weights: w };
}
