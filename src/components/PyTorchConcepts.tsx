"use client";
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Concept {
  title: string;
  icon: string;
  pytorchClass: string;
  description: string;
  code: string;
  math: string;
  visualization: React.ReactNode;
}

const ActivationViz = ({ fn, color }: { fn: string; color: string }) => {
  const points: string[] = [];
  for (let i = -50; i <= 50; i++) {
    const x = i;
    const input = i / 15;
    let y: number;
    if (fn === "relu") y = Math.max(0, input);
    else if (fn === "sigmoid") y = 1 / (1 + Math.exp(-input));
    else y = Math.tanh(input);

    points.push(`${x + 60},${40 - y * 30}`);
  }
  return (
    <svg viewBox="0 0 120 80" className="w-full h-20">
      <line x1="10" y1="40" x2="110" y2="40" stroke="#374151" strokeWidth={0.5} />
      <line x1="60" y1="5" x2="60" y2="75" stroke="#374151" strokeWidth={0.5} />
      <polyline points={points.join(" ")} fill="none" stroke={color} strokeWidth={2} />
      <text x="90" y="75" fill="#6b7280" fontSize={8}>{fn}</text>
    </svg>
  );
};

const GradientFlowViz = () => (
  <svg viewBox="0 0 120 80" className="w-full h-20">
    <defs>
      <marker id="arrowRed" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="#ef4444" />
      </marker>
      <marker id="arrowGreen" markerWidth="6" markerHeight="4" refX="0" refY="2" orient="auto">
        <polygon points="6 0, 0 2, 6 4" fill="#22c55e" />
      </marker>
    </defs>
    {[20, 45, 70, 95].map((x, i) => (
      <circle key={i} cx={x} cy={40} r={10} fill="none" stroke="#6366f1" strokeWidth={1.5} />
    ))}
    <line x1="30" y1="30" x2="85" y2="30" stroke="#22c55e" strokeWidth={1.5} markerEnd="url(#arrowGreen)">
      <animate attributeName="stroke-dashoffset" from="30" to="0" dur="1.5s" repeatCount="indefinite" />
    </line>
    <line x1="85" y1="50" x2="30" y2="50" stroke="#ef4444" strokeWidth={1.5} markerEnd="url(#arrowRed)">
      <animate attributeName="stroke-dashoffset" from="30" to="0" dur="1.5s" repeatCount="indefinite" />
    </line>
    <text x="55" y="22" textAnchor="middle" fill="#22c55e" fontSize={7}>Forward</text>
    <text x="55" y="65" textAnchor="middle" fill="#ef4444" fontSize={7}>Backward</text>
  </svg>
);

const TensorViz = () => (
  <svg viewBox="0 0 120 80" className="w-full h-20">
    {[0, 1, 2].map((row) =>
      [0, 1, 2, 3].map((col) => (
        <g key={`${row}-${col}`}>
          <rect
            x={15 + col * 24}
            y={10 + row * 22}
            width={20}
            height={18}
            fill={`rgba(249, 115, 22, ${0.2 + Math.random() * 0.6})`}
            stroke="#f97316"
            strokeWidth={0.5}
            rx={2}
          />
          <text x={25 + col * 24} y={23 + row * 22} textAnchor="middle" fill="white" fontSize={7} fontFamily="monospace">
            {(Math.random() * 2 - 1).toFixed(1)}
          </text>
        </g>
      ))
    )}
  </svg>
);

const OptimizerViz = () => (
  <svg viewBox="0 0 120 80" className="w-full h-20">
    {/* Loss surface contour */}
    {[25, 20, 15, 10, 5].map((r, i) => (
      <ellipse key={i} cx={75} cy={40} rx={r * 2} ry={r * 1.5} fill="none" stroke={`rgba(99, 102, 241, ${0.15 + i * 0.1})`} strokeWidth={0.8} />
    ))}
    {/* Gradient descent path */}
    <polyline
      points="20,20 35,28 48,33 58,36 65,38 70,39 73,39.5 75,40"
      fill="none"
      stroke="#f97316"
      strokeWidth={1.5}
      strokeDasharray="3,2"
    >
      <animate attributeName="stroke-dashoffset" from="40" to="0" dur="3s" repeatCount="indefinite" />
    </polyline>
    <circle cx="75" cy="40" r={3} fill="#22c55e" />
    <text x="75" y="75" textAnchor="middle" fill="#6b7280" fontSize={7}>minimum</text>
  </svg>
);

const concepts: Concept[] = [
  {
    title: "Tensors",
    icon: "T",
    pytorchClass: "torch.Tensor",
    description:
      "Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with GPU acceleration support and automatic differentiation capabilities.",
    code: `# Creating tensors
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x = torch.randn(3, 4)        # Random normal
x = torch.zeros(2, 3)        # Zeros
x = x.to('cuda')             # GPU transfer
x.requires_grad_(True)       # Enable autograd`,
    math: "T ∈ ℝ^(d₁ × d₂ × ... × dₙ)",
    visualization: <TensorViz />,
  },
  {
    title: "Linear Layer",
    icon: "L",
    pytorchClass: "nn.Linear",
    description:
      "A linear transformation applies a weight matrix multiplication and bias addition. It's the building block of neural networks, connecting layers together.",
    code: `# Linear layer: y = xW^T + b
layer = nn.Linear(in_features=4, out_features=3)
output = layer(input)    # Shape: (batch, 3)

# Access parameters
print(layer.weight.shape)  # (3, 4)
print(layer.bias.shape)    # (3,)`,
    math: "y = xW^T + b, where W ∈ ℝ^(out × in)",
    visualization: (
      <svg viewBox="0 0 120 80" className="w-full h-20">
        {[20, 35, 50, 65].map((y, i) => (
          <circle key={`in-${i}`} cx={20} cy={y} r={6} fill="none" stroke="#3b82f6" strokeWidth={1.5} />
        ))}
        {[28, 45, 62].map((y, i) => (
          <circle key={`out-${i}`} cx={100} cy={y} r={6} fill="none" stroke="#22c55e" strokeWidth={1.5} />
        ))}
        {[20, 35, 50, 65].map((y1) =>
          [28, 45, 62].map((y2) => (
            <line key={`${y1}-${y2}`} x1={26} y1={y1} x2={94} y2={y2} stroke="#6366f1" strokeWidth={0.5} opacity={0.4} />
          ))
        )}
        <text x={60} y={75} textAnchor="middle" fill="#6b7280" fontSize={7}>W @ x + b</text>
      </svg>
    ),
  },
  {
    title: "Activation Functions",
    icon: "σ",
    pytorchClass: "nn.ReLU / nn.Sigmoid / nn.Tanh",
    description:
      "Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. Without them, stacked linear layers would collapse into a single linear transformation.",
    code: `# Activation functions
relu = nn.ReLU()          # max(0, x)
sigmoid = nn.Sigmoid()    # 1 / (1 + e^-x)
tanh = nn.Tanh()          # (e^x - e^-x)/(e^x + e^-x)

# Functional API
import torch.nn.functional as F
out = F.relu(x)`,
    math: "ReLU(x) = max(0, x)\nσ(x) = 1/(1+e^(-x))\ntanh(x) = (e^x - e^(-x))/(e^x + e^(-x))",
    visualization: (
      <div className="flex gap-1">
        <ActivationViz fn="relu" color="#3b82f6" />
        <ActivationViz fn="sigmoid" color="#22c55e" />
        <ActivationViz fn="tanh" color="#f97316" />
      </div>
    ),
  },
  {
    title: "Backpropagation",
    icon: "∇",
    pytorchClass: "loss.backward()",
    description:
      "Backpropagation computes gradients of the loss with respect to all parameters using the chain rule. PyTorch's autograd engine handles this automatically through dynamic computational graphs.",
    code: `# Forward pass
output = model(input)
loss = criterion(output, target)

# Backward pass (computes all gradients)
loss.backward()

# Gradients are stored in .grad
print(model.fc1.weight.grad)`,
    math: "∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w (Chain Rule)",
    visualization: <GradientFlowViz />,
  },
  {
    title: "Optimizer",
    icon: "⟳",
    pytorchClass: "optim.SGD / optim.Adam",
    description:
      "Optimizers update model parameters using computed gradients to minimize the loss function. SGD updates weights proportionally to gradients, while Adam adapts learning rates per-parameter.",
    code: `# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step
optimizer.zero_grad()   # Clear old gradients
loss.backward()         # Compute gradients
optimizer.step()        # Update parameters

# w_new = w_old - lr * gradient`,
    math: "θ_{t+1} = θ_t - η · ∇_θ L(θ_t)",
    visualization: <OptimizerViz />,
  },
  {
    title: "Loss Functions",
    icon: "ℒ",
    pytorchClass: "nn.MSELoss / nn.CrossEntropyLoss",
    description:
      "Loss functions measure how far the model's predictions are from the actual targets. MSE is used for regression, while Cross-Entropy is standard for classification tasks.",
    code: `# Regression loss
mse = nn.MSELoss()
loss = mse(predictions, targets)

# Classification loss
ce = nn.CrossEntropyLoss()
loss = ce(logits, labels)

# Binary classification
bce = nn.BCELoss()
loss = bce(sigmoid_output, binary_target)`,
    math: "MSE = (1/n)Σ(ŷᵢ - yᵢ)²\nCE = -Σ yᵢ log(ŷᵢ)",
    visualization: (
      <svg viewBox="0 0 120 80" className="w-full h-20">
        <line x1="15" y1="65" x2="110" y2="65" stroke="#374151" strokeWidth={0.5} />
        <line x1="15" y1="10" x2="15" y2="65" stroke="#374151" strokeWidth={0.5} />
        {Array.from({ length: 20 }).map((_, i) => {
          const x = 15 + i * 5;
          const y = 60 - Math.pow(i / 20 - 0.5, 2) * 200;
          return <circle key={i} cx={x} cy={y} r={2} fill="#f97316" />;
        })}
        <text x={60} y={78} textAnchor="middle" fill="#6b7280" fontSize={7}>prediction error</text>
        <text x={8} y={40} fill="#6b7280" fontSize={6} transform="rotate(-90 8 40)">loss</text>
      </svg>
    ),
  },
];

export default function PyTorchConcepts() {
  const [selectedConcept, setSelectedConcept] = useState(0);

  return (
    <div className="space-y-4">
      {/* Concept Tabs */}
      <div className="flex flex-wrap gap-2">
        {concepts.map((concept, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedConcept(idx)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedConcept === idx
                ? "bg-orange-600 text-white shadow-lg shadow-orange-600/20"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700"
            }`}
          >
            <span className="text-lg">{concept.icon}</span>
            {concept.title}
          </button>
        ))}
      </div>

      {/* Concept Detail */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedConcept}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="bg-gray-800/50 backdrop-blur rounded-xl border border-gray-700 overflow-hidden"
        >
          <div className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold text-white">{concepts[selectedConcept].title}</h3>
                <code className="text-xs text-orange-400 bg-orange-400/10 px-2 py-0.5 rounded mt-1 inline-block">
                  {concepts[selectedConcept].pytorchClass}
                </code>
              </div>
              <div className="w-32">{concepts[selectedConcept].visualization}</div>
            </div>

            <p className="text-gray-300 text-sm leading-relaxed mb-4">{concepts[selectedConcept].description}</p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-900/60 rounded-lg p-4">
                <h4 className="text-xs font-semibold text-orange-400 mb-2">PyTorch Code</h4>
                <pre className="text-xs font-mono text-green-300 whitespace-pre-wrap leading-relaxed">
                  {concepts[selectedConcept].code}
                </pre>
              </div>
              <div className="bg-gray-900/60 rounded-lg p-4">
                <h4 className="text-xs font-semibold text-blue-400 mb-2">Mathematical Formula</h4>
                <pre className="text-sm font-mono text-blue-300 whitespace-pre-wrap leading-relaxed">
                  {concepts[selectedConcept].math}
                </pre>
                <div className="mt-4">{concepts[selectedConcept].visualization}</div>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
