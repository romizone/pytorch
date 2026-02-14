"use client";
import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";

interface Point {
  x: number;
  y: number;
  label: number;
  predicted?: number;
}

export default function XORPlayground() {
  const [points] = useState<Point[]>([
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 0 },
  ]);

  const [decisionBoundary, setDecisionBoundary] = useState<number[][]>([]);
  const [weights, setWeights] = useState({
    w1: Array.from({ length: 4 }, () => Array.from({ length: 2 }, () => Math.random() * 2 - 1)),
    b1: Array.from({ length: 4 }, () => Math.random() * 0.5 - 0.25),
    w2: Array.from({ length: 1 }, () => Array.from({ length: 4 }, () => Math.random() * 2 - 1)),
    b2: [Math.random() * 0.5 - 0.25],
  });
  const [epoch, setEpoch] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [predictions, setPredictions] = useState<number[]>([0.5, 0.5, 0.5, 0.5]);

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, x))));
  const relu = (x: number) => Math.max(0, x);

  const predict = useCallback(
    (input: number[]) => {
      const h = weights.w1.map((w, i) => relu(w[0] * input[0] + w[1] * input[1] + weights.b1[i]));
      const out = sigmoid(weights.w2[0].reduce((s, w, i) => s + w * h[i], weights.b2[0]));
      return out;
    },
    [weights]
  );

  const trainStep = useCallback(() => {
    const lr = 0.3;
    const newW = JSON.parse(JSON.stringify(weights));

    for (const p of points) {
      const input = [p.x, p.y];
      const h_raw = newW.w1.map((w: number[], i: number) => w[0] * input[0] + w[1] * input[1] + newW.b1[i]);
      const h = h_raw.map(relu);
      const o_raw = newW.w2[0].reduce((s: number, w: number, i: number) => s + w * h[i], newW.b2[0]);
      const o = sigmoid(o_raw);

      const dL_do = o - p.label;
      const do_dz = o * (1 - o);
      const delta_out = dL_do * do_dz;

      for (let i = 0; i < 4; i++) {
        const dh = h_raw[i] > 0 ? 1 : 0;
        const delta_h = delta_out * newW.w2[0][i] * dh;
        newW.w1[i][0] -= lr * delta_h * input[0];
        newW.w1[i][1] -= lr * delta_h * input[1];
        newW.b1[i] -= lr * delta_h;
        newW.w2[0][i] -= lr * delta_out * h[i];
      }
      newW.b2[0] -= lr * delta_out;
    }

    setWeights(newW);
    setEpoch((e) => e + 1);
  }, [weights, points]);

  useEffect(() => {
    if (!isTraining) return;
    const interval = setInterval(trainStep, 50);
    return () => clearInterval(interval);
  }, [isTraining, trainStep]);

  useEffect(() => {
    const preds = points.map((p) => predict([p.x, p.y]));
    setPredictions(preds);

    const res = 20;
    const boundary: number[][] = [];
    for (let i = 0; i <= res; i++) {
      const row: number[] = [];
      for (let j = 0; j <= res; j++) {
        row.push(predict([j / res, i / res]));
      }
      boundary.push(row);
    }
    setDecisionBoundary(boundary);
  }, [weights, predict, points]);

  const gridSize = 250;
  const padding = 30;

  return (
    <div className="bg-gray-800/50 backdrop-blur rounded-xl border border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-white">XOR Decision Boundary</h3>
          <p className="text-xs text-gray-400 mt-1">
            Watch the network learn to classify XOR - a problem that requires non-linear decision boundaries
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400">
            Epoch: <span className="text-orange-400 font-mono">{epoch}</span>
          </span>
          <button
            onClick={() => setIsTraining(!isTraining)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              isTraining ? "bg-red-600 hover:bg-red-700 text-white" : "bg-green-600 hover:bg-green-700 text-white"
            }`}
          >
            {isTraining ? "Stop" : "Train"}
          </button>
          <button
            onClick={() => {
              setWeights({
                w1: Array.from({ length: 4 }, () => Array.from({ length: 2 }, () => Math.random() * 2 - 1)),
                b1: Array.from({ length: 4 }, () => Math.random() * 0.5 - 0.25),
                w2: Array.from({ length: 1 }, () => Array.from({ length: 4 }, () => Math.random() * 2 - 1)),
                b2: [Math.random() * 0.5 - 0.25],
              });
              setEpoch(0);
              setIsTraining(false);
            }}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Decision Boundary Visualization */}
        <div className="flex justify-center">
          <svg width={gridSize + padding * 2} height={gridSize + padding * 2} className="bg-gray-900 rounded-lg">
            {/* Decision boundary heatmap */}
            {decisionBoundary.map((row, i) =>
              row.map((val, j) => {
                const cellSize = gridSize / (decisionBoundary.length - 1);
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={padding + j * cellSize - cellSize / 2}
                    y={padding + (decisionBoundary.length - 1 - i) * cellSize - cellSize / 2}
                    width={cellSize + 1}
                    height={cellSize + 1}
                    fill={val > 0.5 ? `rgba(59, 130, 246, ${val * 0.5})` : `rgba(239, 68, 68, ${(1 - val) * 0.5})`}
                  />
                );
              })
            )}

            {/* Grid lines */}
            <line x1={padding} y1={padding} x2={padding} y2={padding + gridSize} stroke="#4b5563" strokeWidth={1} />
            <line x1={padding} y1={padding + gridSize} x2={padding + gridSize} y2={padding + gridSize} stroke="#4b5563" strokeWidth={1} />

            {/* Axis labels */}
            <text x={padding + gridSize / 2} y={padding + gridSize + 22} textAnchor="middle" fill="#9ca3af" fontSize={11}>
              x₁
            </text>
            <text x={padding - 18} y={padding + gridSize / 2} textAnchor="middle" fill="#9ca3af" fontSize={11} transform={`rotate(-90 ${padding - 18} ${padding + gridSize / 2})`}>
              x₂
            </text>

            {/* Data points */}
            {points.map((p, i) => (
              <g key={i}>
                <motion.circle
                  cx={padding + p.x * gridSize}
                  cy={padding + (1 - p.y) * gridSize}
                  r={12}
                  fill={p.label === 1 ? "#3b82f6" : "#ef4444"}
                  stroke="white"
                  strokeWidth={2}
                  animate={{
                    scale: Math.abs(predictions[i] - p.label) < 0.3 ? [1, 1.15, 1] : 1,
                  }}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
                <text
                  x={padding + p.x * gridSize}
                  y={padding + (1 - p.y) * gridSize + 4}
                  textAnchor="middle"
                  fill="white"
                  fontSize={10}
                  fontWeight="bold"
                >
                  {p.label}
                </text>
              </g>
            ))}

            {/* 0.5 decision boundary line approximation */}
            <text x={padding + 5} y={padding + 12} fill="#6b7280" fontSize={9}>
              (0,1)=1
            </text>
            <text x={padding + gridSize - 35} y={padding + 12} fill="#6b7280" fontSize={9}>
              (1,1)=0
            </text>
            <text x={padding + 5} y={padding + gridSize - 5} fill="#6b7280" fontSize={9}>
              (0,0)=0
            </text>
            <text x={padding + gridSize - 35} y={padding + gridSize - 5} fill="#6b7280" fontSize={9}>
              (1,0)=1
            </text>
          </svg>
        </div>

        {/* Predictions Table */}
        <div className="space-y-4">
          <div className="bg-gray-900/50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-orange-400 mb-3">XOR Truth Table vs Predictions</h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="py-2 text-left">x₁</th>
                  <th className="py-2 text-left">x₂</th>
                  <th className="py-2 text-left">Target</th>
                  <th className="py-2 text-left">Predicted</th>
                  <th className="py-2 text-left">Status</th>
                </tr>
              </thead>
              <tbody>
                {points.map((p, i) => (
                  <tr key={i} className="border-b border-gray-800">
                    <td className="py-2 font-mono">{p.x}</td>
                    <td className="py-2 font-mono">{p.y}</td>
                    <td className="py-2 font-mono">{p.label}</td>
                    <td className="py-2 font-mono text-orange-400">{predictions[i].toFixed(4)}</td>
                    <td className="py-2">
                      {Math.round(predictions[i]) === p.label ? (
                        <span className="text-green-400 text-xs">Correct</span>
                      ) : (
                        <span className="text-red-400 text-xs">Wrong</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900/50 rounded-lg p-4">
            <h4 className="text-xs font-semibold text-orange-400 mb-2">Why XOR needs a Neural Network</h4>
            <p className="text-xs text-gray-400 leading-relaxed">
              XOR cannot be solved by a single linear classifier (perceptron) because the classes are not linearly separable.
              A hidden layer creates intermediate representations that transform the space, making the problem separable.
              This is the fundamental insight behind deep learning.
            </p>
            <div className="mt-2 p-2 bg-gray-800/50 rounded font-mono text-xs text-gray-300">
              <span className="text-orange-400"># The XOR problem requires non-linearity</span>
              <br />
              h = F.relu(self.hidden(x)) &nbsp;<span className="text-gray-500"># Non-linear transform</span>
              <br />
              out = torch.sigmoid(self.output(h))
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
