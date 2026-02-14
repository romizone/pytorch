"use client";
import React, { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Neuron {
  x: number;
  y: number;
  value: number;
  activated: number;
  bias: number;
}

interface Connection {
  from: { layer: number; neuron: number };
  to: { layer: number; neuron: number };
  weight: number;
}

interface NeuralNetworkVisualizerProps {
  onTrainingData?: (data: { epoch: number; loss: number; accuracy: number }) => void;
}

const ACTIVATIONS: Record<string, (x: number) => number> = {
  relu: (x: number) => Math.max(0, x),
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  tanh: (x: number) => Math.tanh(x),
};

export default function NeuralNetworkVisualizer({ onTrainingData }: NeuralNetworkVisualizerProps) {
  const [layers, setLayers] = useState([2, 4, 4, 1]);
  const [neurons, setNeurons] = useState<Neuron[][]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [activation, setActivation] = useState<string>("relu");
  const [forwardPassStep, setForwardPassStep] = useState(-1);
  const [showPyTorchCode, setShowPyTorchCode] = useState(false);
  const [selectedNeuron, setSelectedNeuron] = useState<{ layer: number; neuron: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trainingRef = useRef(false);

  const initializeNetwork = useCallback(() => {
    const newNeurons: Neuron[][] = [];
    const newConnections: Connection[] = [];
    const width = 700;
    const height = 400;

    layers.forEach((count, layerIdx) => {
      const layerNeurons: Neuron[] = [];
      const x = (layerIdx / (layers.length - 1)) * (width - 100) + 50;
      for (let i = 0; i < count; i++) {
        const y = ((i + 1) / (count + 1)) * height;
        layerNeurons.push({
          x,
          y,
          value: Math.random() * 2 - 1,
          activated: 0,
          bias: Math.random() * 0.5 - 0.25,
        });
      }
      newNeurons.push(layerNeurons);
    });

    for (let l = 0; l < layers.length - 1; l++) {
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l + 1]; j++) {
          newConnections.push({
            from: { layer: l, neuron: i },
            to: { layer: l + 1, neuron: j },
            weight: (Math.random() * 2 - 1) * Math.sqrt(2 / layers[l]),
          });
        }
      }
    }

    setNeurons(newNeurons);
    setConnections(newConnections);
    setEpoch(0);
    setLoss(1.0);
    setForwardPassStep(-1);
  }, [layers]);

  useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);

  const forwardPass = useCallback(
    (input: number[]) => {
      if (neurons.length === 0) return;
      const newNeurons = neurons.map((layer) => layer.map((n) => ({ ...n })));
      const activationFn = ACTIVATIONS[activation];

      input.forEach((val, i) => {
        if (newNeurons[0][i]) {
          newNeurons[0][i].value = val;
          newNeurons[0][i].activated = val;
        }
      });

      for (let l = 1; l < newNeurons.length; l++) {
        for (let j = 0; j < newNeurons[l].length; j++) {
          let sum = newNeurons[l][j].bias;
          for (let i = 0; i < newNeurons[l - 1].length; i++) {
            const conn = connections.find(
              (c) => c.from.layer === l - 1 && c.from.neuron === i && c.to.layer === l && c.to.neuron === j
            );
            if (conn) {
              sum += newNeurons[l - 1][i].activated * conn.weight;
            }
          }
          newNeurons[l][j].value = sum;
          newNeurons[l][j].activated = l === newNeurons.length - 1 ? ACTIVATIONS.sigmoid(sum) : activationFn(sum);
        }
      }

      setNeurons(newNeurons);
      return newNeurons[newNeurons.length - 1][0]?.activated ?? 0;
    },
    [neurons, connections, activation]
  );

  const trainStep = useCallback(() => {
    const xorData = [
      { input: [0, 0], target: 0 },
      { input: [0, 1], target: 1 },
      { input: [1, 0], target: 1 },
      { input: [1, 1], target: 0 },
    ];

    let totalLoss = 0;
    let correct = 0;

    const newConnections = connections.map((c) => ({ ...c }));
    const newNeurons = neurons.map((layer) => layer.map((n) => ({ ...n })));

    for (const sample of xorData) {
      const activationFn = ACTIVATIONS[activation];

      sample.input.forEach((val, i) => {
        if (newNeurons[0][i]) {
          newNeurons[0][i].value = val;
          newNeurons[0][i].activated = val;
        }
      });

      for (let l = 1; l < newNeurons.length; l++) {
        for (let j = 0; j < newNeurons[l].length; j++) {
          let sum = newNeurons[l][j].bias;
          for (let i = 0; i < newNeurons[l - 1].length; i++) {
            const conn = newConnections.find(
              (c) => c.from.layer === l - 1 && c.from.neuron === i && c.to.layer === l && c.to.neuron === j
            );
            if (conn) sum += newNeurons[l - 1][i].activated * conn.weight;
          }
          newNeurons[l][j].value = sum;
          newNeurons[l][j].activated = l === newNeurons.length - 1 ? ACTIVATIONS.sigmoid(sum) : activationFn(sum);
        }
      }

      const output = newNeurons[newNeurons.length - 1][0]?.activated ?? 0;
      const error = output - sample.target;
      totalLoss += error * error;
      if (Math.round(output) === sample.target) correct++;

      // Backpropagation (simplified gradient descent)
      for (let l = newNeurons.length - 1; l >= 1; l--) {
        for (let j = 0; j < newNeurons[l].length; j++) {
          const a = newNeurons[l][j].activated;
          let delta: number;
          if (l === newNeurons.length - 1) {
            delta = error * a * (1 - a);
          } else {
            let downstreamError = 0;
            for (let k = 0; k < newNeurons[l + 1].length; k++) {
              const conn = newConnections.find(
                (c) => c.from.layer === l && c.from.neuron === j && c.to.layer === l + 1 && c.to.neuron === k
              );
              if (conn) downstreamError += conn.weight * (newNeurons[l + 1][k].activated * (1 - newNeurons[l + 1][k].activated));
            }
            delta = downstreamError * (activation === "relu" ? (a > 0 ? 1 : 0) : a * (1 - a));
          }

          newNeurons[l][j].bias -= learningRate * delta;

          for (let i = 0; i < newNeurons[l - 1].length; i++) {
            const connIdx = newConnections.findIndex(
              (c) => c.from.layer === l - 1 && c.from.neuron === i && c.to.layer === l && c.to.neuron === j
            );
            if (connIdx !== -1) {
              newConnections[connIdx].weight -= learningRate * delta * newNeurons[l - 1][i].activated;
            }
          }
        }
      }
    }

    setConnections(newConnections);
    setNeurons(newNeurons);
    const avgLoss = totalLoss / xorData.length;
    const acc = correct / xorData.length;
    setLoss(avgLoss);
    setEpoch((e) => e + 1);

    return { loss: avgLoss, accuracy: acc };
  }, [neurons, connections, activation, learningRate]);

  useEffect(() => {
    if (!isTraining) return;
    trainingRef.current = true;

    const interval = setInterval(() => {
      if (!trainingRef.current) return;
      const result = trainStep();
      setEpoch((e) => {
        onTrainingData?.({ epoch: e, loss: result.loss, accuracy: result.accuracy });
        return e;
      });
      if (result.loss < 0.001) {
        setIsTraining(false);
        trainingRef.current = false;
      }
    }, 100);

    return () => {
      clearInterval(interval);
      trainingRef.current = false;
    };
  }, [isTraining, trainStep, onTrainingData]);

  const runForwardPassAnimation = async () => {
    for (let i = 0; i < layers.length; i++) {
      setForwardPassStep(i);
      await new Promise((r) => setTimeout(r, 600));
    }
    forwardPass([1, 0]);
    setForwardPassStep(-1);
  };

  const getWeightColor = (w: number) => {
    if (w > 0) return `rgba(59, 130, 246, ${Math.min(Math.abs(w), 1)})`;
    return `rgba(239, 68, 68, ${Math.min(Math.abs(w), 1)})`;
  };

  const getActivationColor = (v: number) => {
    const intensity = Math.min(Math.abs(v), 1);
    if (v > 0) return `rgba(34, 197, 94, ${0.3 + intensity * 0.7})`;
    return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`;
  };

  const addLayer = () => {
    if (layers.length < 6) {
      const newLayers = [...layers];
      newLayers.splice(layers.length - 1, 0, 3);
      setLayers(newLayers);
    }
  };

  const removeLayer = () => {
    if (layers.length > 3) {
      const newLayers = [...layers];
      newLayers.splice(layers.length - 2, 1);
      setLayers(newLayers);
    }
  };

  const adjustNeurons = (layerIdx: number, delta: number) => {
    if (layerIdx === 0 || layerIdx === layers.length - 1) return;
    const newLayers = [...layers];
    newLayers[layerIdx] = Math.max(1, Math.min(8, newLayers[layerIdx] + delta));
    setLayers(newLayers);
  };

  const pyTorchCode = `import torch
import torch.nn as nn
import torch.optim as optim

# Define Neural Network Architecture
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        ${layers
          .slice(0, -1)
          .map((n, i) => `self.fc${i + 1} = nn.Linear(${n}, ${layers[i + 1]})`)
          .join("\n        ")}
        self.${activation} = nn.${activation === "relu" ? "ReLU" : activation === "sigmoid" ? "Sigmoid" : "Tanh"}()

    def forward(self, x):
        ${layers
          .slice(0, -1)
          .map((_, i) => {
            if (i === layers.length - 2) return `x = torch.sigmoid(self.fc${i + 1}(x))`;
            return `x = self.${activation}(self.fc${i + 1}(x))`;
          })
          .join("\n        ")}
        return x

# Training Setup
model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=${learningRate})

# XOR Dataset
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Training Loop
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")`;

  const layerLabels = ["Input", ...layers.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), "Output"];

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-orange-400 mb-3">Architecture</h3>
          <div className="flex items-center gap-2 mb-2">
            <button onClick={removeLayer} className="px-3 py-1 bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 text-sm">
              - Layer
            </button>
            <span className="text-gray-300 text-sm">{layers.length} layers</span>
            <button onClick={addLayer} className="px-3 py-1 bg-green-600/20 text-green-400 rounded hover:bg-green-600/30 text-sm">
              + Layer
            </button>
          </div>
          <div className="flex flex-wrap gap-1">
            {layers.map((n, i) => (
              <div key={i} className="flex items-center gap-1 bg-gray-700/50 rounded px-2 py-1">
                <span className="text-xs text-gray-400">{layerLabels[i]}</span>
                {i > 0 && i < layers.length - 1 && (
                  <>
                    <button onClick={() => adjustNeurons(i, -1)} className="text-xs text-red-400 hover:text-red-300">
                      -
                    </button>
                    <span className="text-xs text-white font-mono">{n}</span>
                    <button onClick={() => adjustNeurons(i, 1)} className="text-xs text-green-400 hover:text-green-300">
                      +
                    </button>
                  </>
                )}
                {(i === 0 || i === layers.length - 1) && <span className="text-xs text-white font-mono">{n}</span>}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-orange-400 mb-3">Hyperparameters</h3>
          <div className="space-y-2">
            <div>
              <label className="text-xs text-gray-400">Learning Rate: {learningRate.toFixed(3)}</label>
              <input
                type="range"
                min="0.001"
                max="1"
                step="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full accent-orange-500"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Activation Function</label>
              <div className="flex gap-1 mt-1">
                {["relu", "sigmoid", "tanh"].map((fn) => (
                  <button
                    key={fn}
                    onClick={() => setActivation(fn)}
                    className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
                      activation === fn ? "bg-orange-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                    }`}
                  >
                    {fn}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur rounded-xl p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-orange-400 mb-3">Training Controls</h3>
          <div className="space-y-2">
            <div className="flex gap-2">
              <button
                onClick={() => setIsTraining(!isTraining)}
                className={`flex-1 px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
                  isTraining
                    ? "bg-red-600 hover:bg-red-700 text-white"
                    : "bg-green-600 hover:bg-green-700 text-white"
                }`}
              >
                {isTraining ? "⏹ Stop" : "▶ Train"}
              </button>
              <button
                onClick={runForwardPassAnimation}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
              >
                Forward Pass
              </button>
            </div>
            <button
              onClick={initializeNetwork}
              className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm"
            >
              Reset Network
            </button>
          </div>
        </div>
      </div>

      {/* Network Visualization */}
      <div className="bg-gray-900/80 backdrop-blur rounded-xl border border-gray-700 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
          <h3 className="text-sm font-semibold text-white">Neural Network Visualization</h3>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-gray-400">
              Epoch: <span className="text-orange-400 font-mono">{epoch}</span>
            </span>
            <span className="text-gray-400">
              Loss: <span className="text-orange-400 font-mono">{loss.toFixed(6)}</span>
            </span>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-blue-500"></div>
              <span className="text-gray-500">Positive</span>
              <div className="w-3 h-0.5 bg-red-500"></div>
              <span className="text-gray-500">Negative</span>
            </div>
          </div>
        </div>
        <svg viewBox="0 0 700 420" className="w-full" style={{ minHeight: 300 }}>
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#0f172a" />
              <stop offset="100%" stopColor="#1e1b4b" />
            </linearGradient>
          </defs>

          <rect width="700" height="420" fill="url(#bgGrad)" rx="12" />

          {/* Layer Labels */}
          {neurons.map((layer, layerIdx) => (
            <text
              key={`label-${layerIdx}`}
              x={layer[0]?.x ?? 0}
              y={15}
              textAnchor="middle"
              className="text-[10px]"
              fill="#94a3b8"
            >
              {layerLabels[layerIdx]}
            </text>
          ))}

          {/* Connections */}
          {connections.map((conn, idx) => {
            const from = neurons[conn.from.layer]?.[conn.from.neuron];
            const to = neurons[conn.to.layer]?.[conn.to.neuron];
            if (!from || !to) return null;

            const isActive = forwardPassStep >= conn.to.layer;
            return (
              <line
                key={`conn-${idx}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={getWeightColor(conn.weight)}
                strokeWidth={Math.max(0.5, Math.abs(conn.weight) * 2)}
                opacity={isActive && forwardPassStep >= 0 ? 1 : forwardPassStep >= 0 ? 0.15 : 0.6}
                className="transition-all duration-300"
              />
            );
          })}

          {/* Neurons */}
          {neurons.map((layer, layerIdx) =>
            layer.map((neuron, neuronIdx) => {
              const isActive = forwardPassStep >= layerIdx;
              const isSelected = selectedNeuron?.layer === layerIdx && selectedNeuron?.neuron === neuronIdx;

              return (
                <g
                  key={`neuron-${layerIdx}-${neuronIdx}`}
                  onClick={() => setSelectedNeuron(isSelected ? null : { layer: layerIdx, neuron: neuronIdx })}
                  className="cursor-pointer"
                >
                  {/* Glow effect */}
                  {(isActive || isSelected) && (
                    <circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={20}
                      fill={getActivationColor(neuron.activated)}
                      opacity={0.3}
                      filter="url(#glow)"
                    />
                  )}
                  {/* Neuron circle */}
                  <circle
                    cx={neuron.x}
                    cy={neuron.y}
                    r={14}
                    fill={getActivationColor(neuron.activated)}
                    stroke={isSelected ? "#f97316" : isActive && forwardPassStep >= 0 ? "#60a5fa" : "#475569"}
                    strokeWidth={isSelected ? 3 : 2}
                    className="transition-all duration-300"
                  />
                  {/* Value */}
                  <text x={neuron.x} y={neuron.y + 4} textAnchor="middle" fill="white" className="text-[9px] font-mono" fontWeight="bold">
                    {neuron.activated.toFixed(2)}
                  </text>
                </g>
              );
            })
          )}

          {/* Forward pass arrow animation */}
          {forwardPassStep >= 0 && forwardPassStep < neurons.length && neurons[forwardPassStep] && (
            <g>
              <rect
                x={neurons[forwardPassStep][0].x - 35}
                y={neurons[forwardPassStep].length > 0 ? Math.min(...neurons[forwardPassStep].map((n) => n.y)) - 30 : 0}
                width={70}
                height={
                  neurons[forwardPassStep].length > 1
                    ? Math.max(...neurons[forwardPassStep].map((n) => n.y)) -
                      Math.min(...neurons[forwardPassStep].map((n) => n.y)) +
                      60
                    : 60
                }
                fill="none"
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="5,5"
                rx={8}
                opacity={0.6}
              >
                <animate attributeName="stroke-dashoffset" from="0" to="20" dur="1s" repeatCount="indefinite" />
              </rect>
            </g>
          )}
        </svg>
      </div>

      {/* Neuron Details Panel */}
      <AnimatePresence>
        {selectedNeuron && neurons[selectedNeuron.layer]?.[selectedNeuron.neuron] && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gray-800/80 backdrop-blur rounded-xl border border-orange-500/30 p-4"
          >
            <h4 className="text-orange-400 font-semibold text-sm mb-2">
              {layerLabels[selectedNeuron.layer]} - Neuron {selectedNeuron.neuron + 1}
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400">Raw Value (z)</div>
                <div className="text-white font-mono">{neurons[selectedNeuron.layer][selectedNeuron.neuron].value.toFixed(4)}</div>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400">Activated (a)</div>
                <div className="text-white font-mono">
                  {neurons[selectedNeuron.layer][selectedNeuron.neuron].activated.toFixed(4)}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400">Bias (b)</div>
                <div className="text-white font-mono">{neurons[selectedNeuron.layer][selectedNeuron.neuron].bias.toFixed(4)}</div>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400">Formula</div>
                <div className="text-green-400 font-mono">a = {activation}(z + b)</div>
              </div>
            </div>
            <div className="mt-3 p-2 bg-gray-900/50 rounded font-mono text-xs text-gray-300">
              <span className="text-orange-400"># PyTorch equivalent:</span>
              <br />
              z = torch.matmul(W, x) + b &nbsp;&nbsp;<span className="text-gray-500"># Linear transformation</span>
              <br />a = F.{activation}(z) &nbsp;&nbsp;<span className="text-gray-500"># Activation function</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* PyTorch Code */}
      <div className="bg-gray-800/50 backdrop-blur rounded-xl border border-gray-700 overflow-hidden">
        <button
          onClick={() => setShowPyTorchCode(!showPyTorchCode)}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-700/30 transition-colors"
        >
          <span className="text-sm font-semibold text-orange-400">
            PyTorch Code (auto-generated from your network)
          </span>
          <span className="text-gray-400 text-sm">{showPyTorchCode ? "▲" : "▼"}</span>
        </button>
        <AnimatePresence>
          {showPyTorchCode && (
            <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
              <pre className="p-4 text-xs font-mono text-green-300 bg-gray-900/80 overflow-x-auto leading-relaxed">
                {pyTorchCode}
              </pre>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
