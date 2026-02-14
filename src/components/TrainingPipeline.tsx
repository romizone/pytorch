"use client";
import React, { useState } from "react";
import { motion } from "framer-motion";

const steps = [
  {
    title: "1. Define Model",
    pytorchCode: `class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = Net()`,
    description: "Define the network architecture by subclassing nn.Module. The __init__ method creates layers, and forward() defines how data flows through them.",
    icon: "🏗️",
    color: "from-blue-500 to-blue-700",
  },
  {
    title: "2. Prepare Data",
    pytorchCode: `# Create dataset
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]],
                  dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]],
                  dtype=torch.float32)

# DataLoader for batching
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4,
                    shuffle=True)`,
    description: "Convert data to tensors and wrap in DataLoader for efficient batching, shuffling, and parallel loading during training.",
    icon: "📊",
    color: "from-purple-500 to-purple-700",
  },
  {
    title: "3. Set Loss & Optimizer",
    pytorchCode: `# Binary classification loss
criterion = nn.BCELoss()

# Adam optimizer with learning rate
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)`,
    description: "Choose a loss function to measure prediction error and an optimizer to update weights. The optimizer accesses all model parameters automatically.",
    icon: "⚙️",
    color: "from-orange-500 to-orange-700",
  },
  {
    title: "4. Training Loop",
    pytorchCode: `for epoch in range(1000):
    for batch_x, batch_y in loader:
        # Forward pass
        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`,
    description: "The training loop: forward pass computes predictions, loss.backward() computes gradients via backpropagation, and optimizer.step() updates all parameters.",
    icon: "🔄",
    color: "from-green-500 to-green-700",
  },
  {
    title: "5. Evaluate",
    pytorchCode: `# Switch to evaluation mode
model.eval()

with torch.no_grad():
    predictions = model(X_test)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == y_test).sum()
    accuracy /= len(y_test)
    print(f"Accuracy: {accuracy:.2%}")`,
    description: "Switch to eval mode (disables dropout/batchnorm training behavior), disable gradient computation for efficiency, and measure model performance.",
    icon: "📈",
    color: "from-teal-500 to-teal-700",
  },
];

export default function TrainingPipeline() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="bg-gray-800/50 backdrop-blur rounded-xl border border-gray-700 p-6">
      <h3 className="text-lg font-bold text-white mb-2">PyTorch Training Pipeline</h3>
      <p className="text-xs text-gray-400 mb-6">
        Click each step to explore the complete neural network training workflow in PyTorch
      </p>

      {/* Step indicators */}
      <div className="flex items-center justify-between mb-8 relative">
        <div className="absolute top-5 left-0 right-0 h-0.5 bg-gray-700" />
        {steps.map((step, idx) => (
          <button
            key={idx}
            onClick={() => setActiveStep(idx)}
            className="relative z-10 flex flex-col items-center"
          >
            <motion.div
              className={`w-10 h-10 rounded-full flex items-center justify-center text-lg bg-gradient-to-br ${
                idx <= activeStep ? step.color : "from-gray-700 to-gray-800"
              } border-2 ${idx === activeStep ? "border-white" : "border-gray-600"} transition-all`}
              animate={{ scale: idx === activeStep ? 1.15 : 1 }}
            >
              {step.icon}
            </motion.div>
            <span
              className={`text-[10px] mt-2 max-w-[80px] text-center leading-tight ${
                idx === activeStep ? "text-white font-semibold" : "text-gray-500"
              }`}
            >
              {step.title}
            </span>
          </button>
        ))}
      </div>

      {/* Active step content */}
      <motion.div
        key={activeStep}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-4"
      >
        <div className="bg-gray-900/60 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-orange-400 mb-2">PyTorch Code</h4>
          <pre className="text-xs font-mono text-green-300 whitespace-pre-wrap leading-relaxed">
            {steps[activeStep].pytorchCode}
          </pre>
        </div>
        <div className="space-y-4">
          <div className="bg-gray-900/60 rounded-lg p-4">
            <h4 className="text-xs font-semibold text-blue-400 mb-2">Explanation</h4>
            <p className="text-sm text-gray-300 leading-relaxed">{steps[activeStep].description}</p>
          </div>

          {/* Visual flow */}
          <div className="bg-gray-900/60 rounded-lg p-4">
            <h4 className="text-xs font-semibold text-purple-400 mb-2">Data Flow</h4>
            <div className="flex items-center gap-2 flex-wrap">
              {activeStep >= 0 && (
                <span className="px-2 py-1 bg-blue-600/20 text-blue-400 rounded text-xs">Input Tensor</span>
              )}
              {activeStep >= 0 && <span className="text-gray-500">→</span>}
              {activeStep >= 0 && (
                <span className="px-2 py-1 bg-purple-600/20 text-purple-400 rounded text-xs">Model</span>
              )}
              {activeStep >= 2 && <span className="text-gray-500">→</span>}
              {activeStep >= 2 && (
                <span className="px-2 py-1 bg-orange-600/20 text-orange-400 rounded text-xs">Predictions</span>
              )}
              {activeStep >= 3 && <span className="text-gray-500">→</span>}
              {activeStep >= 3 && (
                <span className="px-2 py-1 bg-red-600/20 text-red-400 rounded text-xs">Loss</span>
              )}
              {activeStep >= 3 && <span className="text-gray-500">→</span>}
              {activeStep >= 3 && (
                <span className="px-2 py-1 bg-green-600/20 text-green-400 rounded text-xs">Gradients</span>
              )}
              {activeStep >= 3 && <span className="text-gray-500">→</span>}
              {activeStep >= 3 && (
                <span className="px-2 py-1 bg-teal-600/20 text-teal-400 rounded text-xs">Update Weights</span>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
