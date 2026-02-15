<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Neural_Network-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="Neural Network" />
  <img src="https://img.shields.io/badge/Next.js_16-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel" />
</p>

<h1 align="center">
  <br>
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.svg" alt="PyTorch Logo" width="120">
  <br>
  PyTorch Neural Network Simulation
  <br>
</h1>

<h4 align="center">An interactive web-based platform for learning neural network fundamentals through real-time simulations with auto-generated PyTorch code.</h4>

<p align="center">
  <a href="https://pytorch-ecru.vercel.app">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-pytorch--ecru.vercel.app-orange?style=for-the-badge" alt="Live Demo" />
  </a>
  <a href="https://pytorch-ecru.vercel.app/paper">
    <img src="https://img.shields.io/badge/📄_Research_Paper-ArXiv_Style-red?style=for-the-badge" alt="Paper" />
  </a>
  <a href="https://pytorch-ecru.vercel.app/paper/knight-chess">
    <img src="https://img.shields.io/badge/♞_Chess_AI_Paper-Knight_Chess-blue?style=for-the-badge" alt="Chess Paper" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/romizone/pytorch?style=flat-square&color=blue" alt="License" />
  <img src="https://img.shields.io/github/stars/romizone/pytorch?style=flat-square&color=yellow" alt="Stars" />
  <img src="https://img.shields.io/github/forks/romizone/pytorch?style=flat-square&color=green" alt="Forks" />
  <img src="https://img.shields.io/github/last-commit/romizone/pytorch?style=flat-square&color=purple" alt="Last Commit" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome" />
</p>

---

## 📸 Preview

<p align="center">
  <img src="https://img.shields.io/badge/⚡_Live_Simulation-Neural_Network_Visualizer-FF6F00?style=for-the-badge" />
</p>

| Feature | Description |
|:---:|:---|
| 🧠 **Neural Network Visualizer** | Interactive SVG network with real-time training, neuron inspection, and weight visualization |
| 📊 **Training Charts** | Live loss curves and accuracy metrics with PyTorch code annotations |
| 🎯 **XOR Playground** | Decision boundary heatmap showing non-linear classification in action |
| 📚 **PyTorch Concepts** | Interactive guide to Tensors, Layers, Activations, Backprop, Optimizers, and Loss Functions |
| 🔧 **Training Pipeline** | Step-by-step walkthrough of the complete PyTorch training workflow |

---

## ✨ Features

### 🧠 Live Neural Network Simulation
- **Configurable architecture** — 2 to 6 layers, 1 to 8 neurons per hidden layer
- **Real-time training** with live weight updates and neuron activation visualization
- **Forward pass animation** — watch data flow layer by layer
- **Click any neuron** to inspect pre-activation (z), activation (a), bias (b), and formula
- **Weight visualization** — color (blue/red) and thickness encode weight values

### ⚙️ Hyperparameter Controls
- **Learning rate** slider (0.001 – 1.0)
- **Activation functions** — ReLU, Sigmoid, Tanh (switch in real-time)
- **Architecture modification** — add/remove layers and neurons dynamically

### 🔥 Auto-Generated PyTorch Code
Every change to the network architecture automatically generates valid PyTorch code:
```python
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
```

### 📊 Real-Time Training Metrics
- **Loss curve** (MSE) with area chart visualization
- **Accuracy tracking** with percentage display
- PyTorch code annotations on every chart

### 🎯 XOR Decision Boundary
- 2D heatmap that updates during training
- Watch the network learn non-linear classification
- Truth table comparison (target vs. predicted)

### 📄 ArXiv-Style Research Papers
- **Neural Network Paper** — Full paper on the simulation platform with equations, figures, and references
- **Knight Chess AI Paper** — Analysis of chess AI with minimax, alpha-beta pruning, and complexity analysis

---

## 🛠️ Tech Stack

| Technology | Purpose |
|:---:|:---|
| <img src="https://img.shields.io/badge/Next.js-000?logo=next.js&logoColor=white" /> | React framework with SSR and routing |
| <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white" /> | Type-safe development |
| <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?logo=tailwindcss&logoColor=white" /> | Utility-first styling |
| <img src="https://img.shields.io/badge/Framer_Motion-0055FF?logo=framer&logoColor=white" /> | Smooth animations and transitions |
| <img src="https://img.shields.io/badge/Recharts-22B5BF?logo=chart.js&logoColor=white" /> | Training metrics visualization |
| <img src="https://img.shields.io/badge/Vercel-000?logo=vercel&logoColor=white" /> | Edge deployment and hosting |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+
- **npm** or **yarn**

### Installation

```bash
# Clone the repository
git clone https://github.com/romizone/pytorch.git

# Navigate to the project
cd pytorch

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

---

## 📁 Project Structure

```
pytorch/
├── src/
│   ├── app/
│   │   ├── page.tsx                    # Main simulation page
│   │   ├── layout.tsx                  # Root layout
│   │   ├── globals.css                 # Global styles
│   │   └── paper/
│   │       ├── page.tsx                # Neural Network paper (ArXiv style)
│   │       └── knight-chess/
│   │           └── page.tsx            # Knight Chess AI paper (ArXiv style)
│   └── components/
│       ├── NeuralNetworkVisualizer.tsx  # Core network simulation
│       ├── TrainingChart.tsx            # Loss & accuracy charts
│       ├── PyTorchConcepts.tsx          # Interactive concept explorer
│       ├── XORPlayground.tsx            # XOR decision boundary
│       └── TrainingPipeline.tsx         # Training workflow guide
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.ts
```

---

## 📄 Research Papers

### 📑 Neural Network Simulation Paper
> **arXiv:2602.09847v1 [cs.LG]** — 14 Feb 2026

Full academic paper analyzing the simulation platform:
- System architecture and component design
- Neural network engine (forward prop, backprop, gradient descent)
- 10 numbered equations, 3 figures, 3 tables
- 10 academic references

🔗 [**Read Paper →**](https://pytorch-ecru.vercel.app/paper)

### ♞ Knight Chess AI Paper
> **arXiv:2602.10234v1 [cs.AI]** — 14 Feb 2026

Analysis of the Knight Chess game AI:
- Game design: 8×9 board, 5 knights, 3,136 starting positions
- AI engine: Minimax + Alpha-Beta pruning (3 difficulty levels)
- Complexity analysis: branching factor ~42, game tree ~10¹³⁰
- 10 equations, 3 figures, 7 tables, 12 references

🔗 [**Read Paper →**](https://pytorch-ecru.vercel.app/paper/knight-chess)

---

## 🧮 Core Algorithms

### Forward Propagation
```
z_j^(l) = Σ w_ij · a_i^(l-1) + b_j^(l)
a_j^(l) = σ(z_j^(l))
```

### Backpropagation
```
δ_out = (ŷ - y) · σ'(z)
w ← w - η · δ · a_prev
```

### Supported Activation Functions
| Function | Formula | Best For |
|:---:|:---:|:---|
| **ReLU** | max(0, x) | Hidden layers (fastest convergence) |
| **Sigmoid** | 1/(1+e⁻ˣ) | Output layer (binary classification) |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Hidden layers (centered output) |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

<p align="center">
  <strong>Romin Urismanto</strong>
  <br>
  <a href="https://github.com/romizone">
    <img src="https://img.shields.io/badge/GitHub-romizone-181717?style=for-the-badge&logo=github" alt="GitHub" />
  </a>
</p>

---

<p align="center">
  <strong>⭐ Star this repo if you found it helpful!</strong>
  <br><br>
  <img src="https://img.shields.io/badge/Made_with-❤️-red?style=for-the-badge" alt="Made with love" />
  <img src="https://img.shields.io/badge/Powered_by-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="Powered by PyTorch" />
</p>
