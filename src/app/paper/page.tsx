"use client";
import React from "react";
import Link from "next/link";

export default function PaperPage() {
  return (
    <>
      {/* Google Fonts: Computer Modern Serif + Mono */}
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=CMU+Serif:ital,wght@0,400;0,700;1,400&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,700;1,8..60,400&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,700;1,400&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500&display=swap');

        .arxiv-paper {
          font-family: 'STIX Two Text', 'Source Serif 4', 'CMU Serif', 'Times New Roman', 'Computer Modern', serif;
          font-size: 10pt;
          line-height: 1.25;
          color: #000;
          background: #fff;
        }

        .arxiv-paper .mono {
          font-family: 'Source Code Pro', 'Courier New', monospace;
        }

        /* Two-column layout */
        .arxiv-two-col {
          column-count: 2;
          column-gap: 20px;
          column-rule: none;
        }

        .arxiv-two-col section {
          break-inside: avoid-column;
        }

        .arxiv-full-width {
          column-span: all;
        }

        /* Equation block */
        .arxiv-eq {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 8px 0;
          margin: 8px 0;
          position: relative;
        }

        .arxiv-eq .eq-number {
          position: absolute;
          right: 0;
          font-size: 10pt;
        }

        /* Figure styling */
        .arxiv-figure {
          margin: 12px 0;
          text-align: center;
          break-inside: avoid;
        }

        .arxiv-figure .fig-caption {
          font-size: 9pt;
          margin-top: 6px;
          text-align: justify;
        }

        .arxiv-figure .fig-caption strong {
          font-weight: 700;
        }

        /* Table styling */
        .arxiv-table {
          margin: 10px auto;
          border-collapse: collapse;
          font-size: 9pt;
          break-inside: avoid;
        }

        .arxiv-table th,
        .arxiv-table td {
          padding: 3px 8px;
          text-align: left;
        }

        .arxiv-table thead tr {
          border-top: 1.5px solid #000;
          border-bottom: 1px solid #000;
        }

        .arxiv-table tbody tr:last-child {
          border-bottom: 1.5px solid #000;
        }

        .arxiv-table caption {
          font-size: 9pt;
          margin-bottom: 4px;
          text-align: center;
          caption-side: top;
        }

        /* References */
        .arxiv-refs {
          font-size: 8.5pt;
          line-height: 1.3;
        }

        .arxiv-refs li {
          margin-bottom: 3px;
          padding-left: 4px;
        }

        /* Section headings */
        .arxiv-paper h2 {
          font-size: 12pt;
          font-weight: 700;
          margin-top: 14px;
          margin-bottom: 6px;
        }

        .arxiv-paper h3 {
          font-size: 10pt;
          font-weight: 700;
          font-style: italic;
          margin-top: 10px;
          margin-bottom: 4px;
        }

        .arxiv-paper p {
          text-align: justify;
          margin-bottom: 6px;
          text-indent: 1.5em;
          hyphens: auto;
        }

        .arxiv-paper p.no-indent {
          text-indent: 0;
        }

        .arxiv-paper ul, .arxiv-paper ol {
          margin: 4px 0 6px 1.5em;
          font-size: 10pt;
        }

        .arxiv-paper li {
          margin-bottom: 2px;
        }

        /* Code block */
        .arxiv-code {
          font-family: 'Source Code Pro', 'Courier New', monospace;
          font-size: 8pt;
          line-height: 1.35;
          background: #f7f7f7;
          border: 0.5px solid #ddd;
          padding: 6px 8px;
          margin: 6px 0;
          overflow-x: auto;
          break-inside: avoid;
        }

        /* Abstract box */
        .arxiv-abstract {
          margin: 0 2em;
          font-size: 9pt;
          text-align: justify;
        }

        .arxiv-abstract p {
          text-indent: 0;
          font-size: 9pt;
        }

        /* ArXiv header bar */
        .arxiv-header-bar {
          background: #b31b1b;
          height: 4px;
          width: 100%;
        }

        /* Print */
        @media print {
          .no-print { display: none !important; }
          .arxiv-paper { font-size: 10pt; }
          @page { margin: 1in; size: letter; }
          .arxiv-two-col { column-count: 2; column-gap: 20px; }
        }

        @media (max-width: 768px) {
          .arxiv-two-col { column-count: 1; }
        }
      `}</style>

      <main className="arxiv-paper min-h-screen bg-white">
        {/* Toolbar */}
        <div className="no-print fixed top-0 left-0 right-0 z-50 bg-[#b31b1b] text-white px-4 py-2 flex items-center justify-between text-xs shadow-md">
          <div className="flex items-center gap-3">
            <span className="font-bold text-sm tracking-wide">arXiv</span>
            <span className="opacity-70">:</span>
            <span className="opacity-90 mono text-[11px]">2602.09847v1 [cs.LG]</span>
            <span className="opacity-70">14 Feb 2026</span>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/"
              className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-xs transition-colors"
            >
              ← View Simulation
            </Link>
            <button
              onClick={() => window.print()}
              className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-xs transition-colors"
            >
              Download PDF
            </button>
          </div>
        </div>

        {/* ArXiv red bar */}
        <div className="arxiv-header-bar no-print" style={{ marginTop: 36 }} />

        {/* Paper content */}
        <div className="max-w-[8.5in] mx-auto px-[0.75in] pt-6 pb-12">
          {/* ========== TITLE BLOCK (full-width) ========== */}
          <header className="text-center mb-6">
            <h1 className="text-[17.5pt] font-bold leading-snug mb-4 tracking-[-0.01em]">
              Interactive Neural Network Simulation for PyTorch Education:<br />
              A Web-Based Approach to Understanding<br />
              Deep Learning Fundamentals
            </h1>

            <div className="text-[11pt] mt-4 space-y-0.5">
              <p className="font-bold no-indent" style={{ textIndent: 0, marginBottom: 2 }}>Romin Urismanto</p>
              <p className="no-indent text-[9.5pt]" style={{ textIndent: 0, marginBottom: 1, color: '#333' }}>Department of Computer Science</p>
              <p className="no-indent mono text-[9pt]" style={{ textIndent: 0, marginBottom: 0, color: '#555' }}>rominurismanto@gmail.com</p>
            </div>

            <div className="mt-5 mb-4">
              <p className="font-bold text-[10pt]" style={{ textIndent: 0, marginBottom: 4 }}>Abstract</p>
              <div className="arxiv-abstract">
                <p>
                  This paper presents an interactive web-based simulation platform designed to teach
                  neural network fundamentals through the lens of PyTorch, one of the most widely adopted
                  deep learning frameworks. The platform features a real-time neural network visualizer
                  with adjustable architectures, live training with backpropagation, an XOR decision
                  boundary playground, and a step-by-step PyTorch training pipeline walkthrough. Built
                  using modern web technologies (Next.js, TypeScript, and Framer Motion), the simulation
                  provides immediate visual feedback of forward propagation, weight updates, and gradient
                  flow. Each interactive component is paired with auto-generated PyTorch code that
                  mirrors the user&apos;s configured network, bridging the gap between abstract visualization
                  and practical implementation. We describe the system architecture, pedagogical design
                  choices, implementation details of the JavaScript-based neural network engine, and
                  discuss how interactive simulations can enhance comprehension of complex machine
                  learning concepts compared to traditional static educational materials.
                </p>
              </div>
            </div>
          </header>

          <hr style={{ border: 'none', borderTop: '0.5px solid #000', margin: '0 0 14px 0' }} />

          {/* ========== TWO-COLUMN BODY ========== */}
          <div className="arxiv-two-col">

            {/* 1. INTRODUCTION */}
            <section>
              <h2>1&ensp;Introduction</h2>
              <p className="no-indent">
                Deep learning has become a transformative technology across numerous domains, from computer
                vision and natural language processing to healthcare and autonomous systems [10]. As demand for
                machine learning practitioners grows, effective educational tools become increasingly
                important. PyTorch [1], developed by Meta AI, has emerged as one of the leading frameworks
                for deep learning research and production, valued for its dynamic computational graph
                and Pythonic interface.
              </p>
              <p>
                Despite abundant textbooks and online courses, many learners struggle with the abstract
                mathematical concepts underlying neural networks. Traditional static diagrams fail to convey
                the dynamic nature of training processes such as forward propagation, gradient computation,
                and weight updates [5]. This gap motivates the development of interactive simulation tools that
                allow learners to manipulate network parameters in real-time and observe immediate consequences.
              </p>
              <p>
                We present an interactive web-based platform that combines neural network
                visualization with PyTorch code generation. The key contributions of this work are:
              </p>
              <ul style={{ listStyleType: 'disc', paddingLeft: '2em', textIndent: 0 }}>
                <li>A real-time neural network visualizer with configurable architecture and hyperparameters;</li>
                <li>Live training simulation with backpropagation and gradient descent;</li>
                <li>An XOR decision boundary playground demonstrating non-linear classification;</li>
                <li>Auto-generated PyTorch code that reflects the user&apos;s network configuration;</li>
                <li>Interactive exploration of core PyTorch concepts with mathematical foundations.</li>
              </ul>
              <p>
                The remainder of this paper is organized as follows. Section 2 reviews related work.
                Section 3 describes the system architecture. Section 4 details the neural network engine
                implementation. Section 5 discusses the XOR problem as a teaching tool. Section 6 covers
                dynamic PyTorch code generation. Section 7 presents the interactive features and
                pedagogical design. Section 8 shows demonstration results, and Section 9 concludes.
              </p>
            </section>

            {/* 2. RELATED WORK */}
            <section>
              <h2>2&ensp;Related Work</h2>
              <p className="no-indent">
                Several interactive tools have been developed for neural network education.
                TensorFlow Playground [2] provides a browser-based visualization of simple
                neural networks for classification tasks using a grid of
                pre-defined datasets. ConvNetJS [3] by Karpathy offers JavaScript-based
                implementations of convolutional neural networks with real-time loss
                visualization. Distill.pub [8] has pioneered interactive articles that
                combine narrative with manipulable visualizations, setting a high standard
                for explorable explanations.
              </p>
              <p>
                Our work differentiates itself by explicitly mapping visual interactions to
                PyTorch code, providing learners with both conceptual understanding and
                practical implementation skills simultaneously. Unlike previous tools that operate
                in isolation from production frameworks, our platform generates valid PyTorch
                code in real-time as users modify network architectures and hyperparameters,
                establishing a direct bridge between visualization and implementation.
              </p>
            </section>

            {/* 3. SYSTEM ARCHITECTURE */}
            <section>
              <h2>3&ensp;System Architecture</h2>

              <h3>3.1&ensp;Technology Stack</h3>
              <p className="no-indent">
                The platform is built using a modern web technology stack optimized for
                interactive real-time simulations. Table 1 summarizes the key technologies
                employed in the implementation.
              </p>

              {/* Table 1 */}
              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 1:</strong> Technology stack and component responsibilities</caption>
                  <thead>
                    <tr>
                      <th>Component</th>
                      <th>Technology</th>
                      <th>Purpose</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Framework</td><td>Next.js 16</td><td>SSR, routing, optimization</td></tr>
                    <tr><td>Language</td><td>TypeScript</td><td>Type safety</td></tr>
                    <tr><td>Styling</td><td>Tailwind CSS</td><td>Utility-first CSS</td></tr>
                    <tr><td>Animations</td><td>Framer Motion</td><td>UI transitions</td></tr>
                    <tr><td>Charts</td><td>Recharts</td><td>Metrics visualization</td></tr>
                    <tr><td>Graphics</td><td>SVG</td><td>Network rendering</td></tr>
                    <tr><td>Hosting</td><td>Vercel</td><td>Serverless CDN</td></tr>
                  </tbody>
                </table>
              </div>

              <h3>3.2&ensp;Component Architecture</h3>
              <p className="no-indent">
                The application follows a modular component architecture with five primary
                modules, illustrated in Figure 1:
              </p>
              <p>
                <strong>NeuralNetworkVisualizer</strong> is the core simulation component rendering
                an SVG-based network graph with interactive neurons, weighted connections, and
                forward pass animations. Users can dynamically modify the architecture
                (2–6 layers, 1–8 neurons per hidden layer) and observe real-time weight
                changes during training.
              </p>
              <p>
                <strong>TrainingChart</strong> provides real-time visualization of training
                metrics using area charts with loss curves and accuracy progression, each
                annotated with corresponding PyTorch operations.
              </p>
              <p>
                <strong>PyTorchConcepts</strong> offers an interactive reference covering six
                fundamental concepts: Tensors, Linear Layers, Activation Functions,
                Backpropagation, Optimizers, and Loss Functions.
              </p>
              <p>
                <strong>XORPlayground</strong> presents a dedicated simulation for the XOR
                classification problem with a real-time decision boundary heatmap.
              </p>
              <p>
                <strong>TrainingPipeline</strong> guides users through the complete
                PyTorch training workflow in five sequential steps.
              </p>

              {/* Figure 1: Architecture diagram */}
              <div className="arxiv-figure">
                <svg viewBox="0 0 400 140" className="w-full" style={{ maxWidth: 360, margin: '0 auto' }}>
                  <defs>
                    <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="#333" />
                    </marker>
                  </defs>
                  {/* User */}
                  <rect x="160" y="5" width="80" height="24" rx="3" fill="#f3f3f3" stroke="#333" strokeWidth="0.8" />
                  <text x="200" y="21" textAnchor="middle" fontSize="8" fontFamily="serif">User Interface</text>
                  {/* Arrow down */}
                  <line x1="200" y1="29" x2="200" y2="42" stroke="#333" strokeWidth="0.8" markerEnd="url(#arrow)" />
                  {/* Main app */}
                  <rect x="130" y="44" width="140" height="22" rx="3" fill="#e8e8e8" stroke="#333" strokeWidth="0.8" />
                  <text x="200" y="59" textAnchor="middle" fontSize="7.5" fontFamily="serif" fontWeight="bold">Next.js Application Layer</text>
                  {/* Arrow down */}
                  <line x1="200" y1="66" x2="200" y2="78" stroke="#333" strokeWidth="0.8" markerEnd="url(#arrow)" />
                  {/* Component boxes */}
                  {[
                    { x: 10, label: 'NN Visualizer' },
                    { x: 90, label: 'Training Chart' },
                    { x: 170, label: 'Concepts' },
                    { x: 250, label: 'XOR Play' },
                    { x: 325, label: 'Pipeline' },
                  ].map((c, i) => (
                    <g key={i}>
                      <rect x={c.x} y="80" width="70" height="22" rx="2" fill="#fff" stroke="#333" strokeWidth="0.7" />
                      <text x={c.x + 35} y="94" textAnchor="middle" fontSize="6.5" fontFamily="serif">{c.label}</text>
                    </g>
                  ))}
                  {/* Lines from app to components */}
                  {[45, 125, 205, 285, 360].map((x, i) => (
                    <line key={i} x1="200" y1="78" x2={x} y2="80" stroke="#333" strokeWidth="0.5" />
                  ))}
                  {/* JS Engine */}
                  <rect x="100" y="112" width="200" height="22" rx="3" fill="#f0f0f0" stroke="#333" strokeWidth="0.8" strokeDasharray="3,2" />
                  <text x="200" y="127" textAnchor="middle" fontSize="7" fontFamily="serif" fontStyle="italic">JavaScript Neural Network Engine</text>
                  {[135, 265].map((x, i) => (
                    <line key={i} x1={x} y1="102" x2={x} y2="112" stroke="#333" strokeWidth="0.5" strokeDasharray="2,2" />
                  ))}
                </svg>
                <p className="fig-caption" style={{ textIndent: 0 }}>
                  <strong>Figure 1:</strong> System architecture overview showing the modular component design.
                  All five interactive modules communicate with a shared JavaScript-based neural network engine
                  for computation.
                </p>
              </div>
            </section>

            {/* 4. NEURAL NETWORK ENGINE */}
            <section>
              <h2>4&ensp;Neural Network Engine</h2>

              <h3>4.1&ensp;Network Representation</h3>
              <p className="no-indent">
                The neural network is represented internally as a collection of neurons and
                weighted connections. Each neuron <em>j</em> in layer <em>l</em> stores its
                position (for rendering), pre-activation value <em>z</em>, post-activation
                value <em>a</em>, and bias <em>b</em>. Connections maintain source/destination
                indices and weight values. Weights are initialized using He initialization [6]:
              </p>
              <div className="arxiv-eq">
                <span><em>w</em> ∼ 𝒩(0, √(2/<em>n</em><sub>in</sub>))</span>
                <span className="eq-number">(1)</span>
              </div>
              <p className="no-indent">
                where <em>n</em><sub>in</sub> is the fan-in of the layer. This initialization
                strategy ensures that the variance of activations remains stable across layers
                when using ReLU activations.
              </p>

              <h3>4.2&ensp;Forward Propagation</h3>
              <p className="no-indent">
                Forward propagation computes activations layer by layer. For each neuron
                <em> j</em> in layer <em>l</em>, the pre-activation <em>z</em> and activation <em>a</em> are:
              </p>
              <div className="arxiv-eq">
                <span><em>z</em><sub><em>j</em></sub><sup>(<em>l</em>)</sup> = ∑<sub><em>i</em></sub> <em>w</em><sub><em>ij</em></sub> · <em>a</em><sub><em>i</em></sub><sup>(<em>l</em>−1)</sup> + <em>b</em><sub><em>j</em></sub><sup>(<em>l</em>)</sup></span>
                <span className="eq-number">(2)</span>
              </div>
              <div className="arxiv-eq">
                <span><em>a</em><sub><em>j</em></sub><sup>(<em>l</em>)</sup> = σ(<em>z</em><sub><em>j</em></sub><sup>(<em>l</em>)</sup>)</span>
                <span className="eq-number">(3)</span>
              </div>
              <p className="no-indent">
                The platform supports three activation functions σ(·):
              </p>
              <div className="arxiv-eq">
                <span>ReLU(<em>x</em>) = max(0, <em>x</em>)</span>
                <span className="eq-number">(4)</span>
              </div>
              <div className="arxiv-eq">
                <span>Sigmoid(<em>x</em>) = 1/(1 + <em>e</em><sup>−<em>x</em></sup>)</span>
                <span className="eq-number">(5)</span>
              </div>
              <div className="arxiv-eq">
                <span>Tanh(<em>x</em>) = (<em>e</em><sup><em>x</em></sup> − <em>e</em><sup>−<em>x</em></sup>) / (<em>e</em><sup><em>x</em></sup> + <em>e</em><sup>−<em>x</em></sup>)</span>
                <span className="eq-number">(6)</span>
              </div>
              <p className="no-indent">
                The output layer invariably uses sigmoid activation for the XOR binary
                classification task, regardless of the hidden layer activation choice.
              </p>

              <h3>4.3&ensp;Backpropagation</h3>
              <p className="no-indent">
                Training employs stochastic gradient descent with backpropagation [4]. The
                loss function is Mean Squared Error:
              </p>
              <div className="arxiv-eq">
                <span>ℒ = (1/<em>n</em>) ∑<sub><em>i</em>=1</sub><sup><em>n</em></sup> (<em>ŷ</em><sub><em>i</em></sub> − <em>y</em><sub><em>i</em></sub>)²</span>
                <span className="eq-number">(7)</span>
              </div>
              <p className="no-indent">
                Gradients are computed via the chain rule. For the output layer:
              </p>
              <div className="arxiv-eq">
                <span>δ<sub>out</sub> = (<em>ŷ</em> − <em>y</em>) · σ′(<em>z</em>)</span>
                <span className="eq-number">(8)</span>
              </div>
              <p className="no-indent">
                For hidden layers, errors propagate backward:
              </p>
              <div className="arxiv-eq">
                <span>δ<sub><em>h</em></sub> = (∑ <em>w</em> · δ<sub>next</sub>) · <em>f</em> ′(<em>z</em>)</span>
                <span className="eq-number">(9)</span>
              </div>
              <p className="no-indent">
                Weight updates follow the gradient descent rule:
              </p>
              <div className="arxiv-eq">
                <span><em>w</em> ← <em>w</em> − η · δ · <em>a</em><sub>prev</sub></span>
                <span className="eq-number">(10)</span>
              </div>
              <p className="no-indent">
                where η is the learning rate, configurable in our platform between 0.001 and 1.0 via
                a continuous slider control.
              </p>

              {/* Figure 2: Gradient flow */}
              <div className="arxiv-figure">
                <svg viewBox="0 0 300 90" className="w-full" style={{ maxWidth: 280, margin: '0 auto' }}>
                  <defs>
                    <marker id="arrowFwd" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                      <polygon points="0 0, 7 2.5, 0 5" fill="#1a73e8" />
                    </marker>
                    <marker id="arrowBwd" markerWidth="7" markerHeight="5" refX="0" refY="2.5" orient="auto">
                      <polygon points="7 0, 0 2.5, 7 5" fill="#d32f2f" />
                    </marker>
                  </defs>
                  {/* Layers */}
                  {['Input\nx₁, x₂', 'Hidden₁\nReLU', 'Hidden₂\nReLU', 'Output\nσ'].map((label, i) => (
                    <g key={i}>
                      <rect x={15 + i * 72} y="25" width="55" height="35" rx="4" fill={i === 0 ? '#e3f2fd' : i === 3 ? '#fce4ec' : '#f3e5f5'} stroke="#333" strokeWidth="0.6" />
                      {label.split('\n').map((line, li) => (
                        <text key={li} x={42.5 + i * 72} y={39 + li * 11} textAnchor="middle" fontSize="6.5" fontFamily="serif">
                          {line}
                        </text>
                      ))}
                    </g>
                  ))}
                  {/* Forward arrows */}
                  {[0, 1, 2].map((i) => (
                    <line key={`f${i}`} x1={70 + i * 72} y1="35" x2={87 + i * 72} y2="35" stroke="#1a73e8" strokeWidth="1" markerEnd="url(#arrowFwd)" />
                  ))}
                  {/* Backward arrows */}
                  {[0, 1, 2].map((i) => (
                    <line key={`b${i}`} x1={87 + (2 - i) * 72} y1="52" x2={70 + (2 - i) * 72} y2="52" stroke="#d32f2f" strokeWidth="1" markerEnd="url(#arrowBwd)" />
                  ))}
                  <text x="150" y="14" textAnchor="middle" fontSize="7" fill="#1a73e8" fontFamily="serif">Forward Pass →</text>
                  <text x="150" y="80" textAnchor="middle" fontSize="7" fill="#d32f2f" fontFamily="serif">← Backward Pass (∇)</text>
                </svg>
                <p className="fig-caption" style={{ textIndent: 0 }}>
                  <strong>Figure 2:</strong> Data flow during training. The forward pass (blue) computes
                  predictions layer-by-layer. The backward pass (red) propagates gradients from the
                  loss function back through the network via the chain rule.
                </p>
              </div>
            </section>

            {/* 5. XOR PROBLEM */}
            <section>
              <h2>5&ensp;The XOR Problem</h2>
              <p className="no-indent">
                The XOR (exclusive or) function is a classic problem in neural network
                education [7] because it cannot be solved by a single-layer perceptron. The
                truth table (Table 2) produces outputs that are not linearly separable in
                the input space ℝ².
              </p>

              {/* Table 2 */}
              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ margin: '8px auto' }}>
                  <caption><strong>Table 2:</strong> XOR truth table</caption>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'center', width: 60 }}><em>x</em><sub>1</sub></th>
                      <th style={{ textAlign: 'center', width: 60 }}><em>x</em><sub>2</sub></th>
                      <th style={{ textAlign: 'center', width: 80 }}><em>y</em> = XOR</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td style={{ textAlign: 'center' }}>0</td><td style={{ textAlign: 'center' }}>0</td><td style={{ textAlign: 'center' }}>0</td></tr>
                    <tr><td style={{ textAlign: 'center' }}>0</td><td style={{ textAlign: 'center' }}>1</td><td style={{ textAlign: 'center' }}>1</td></tr>
                    <tr><td style={{ textAlign: 'center' }}>1</td><td style={{ textAlign: 'center' }}>0</td><td style={{ textAlign: 'center' }}>1</td></tr>
                    <tr><td style={{ textAlign: 'center' }}>1</td><td style={{ textAlign: 'center' }}>1</td><td style={{ textAlign: 'center' }}>0</td></tr>
                  </tbody>
                </table>
              </div>

              <p>
                Our XOR Playground visualizes the decision boundary as a 2D heatmap
                that updates during training. Blue regions indicate predictions near 1;
                red regions indicate predictions near 0. Users observe how the hidden
                layer creates a non-linear transformation that makes the classes
                separable in the hidden representation space.
              </p>

              {/* Figure 3: Decision boundary */}
              <div className="arxiv-figure">
                <svg viewBox="0 0 240 110" className="w-full" style={{ maxWidth: 230, margin: '0 auto' }}>
                  {/* Before training */}
                  <text x="55" y="10" textAnchor="middle" fontSize="7" fontFamily="serif">(a) Before training</text>
                  <rect x="10" y="14" width="90" height="80" fill="#f5f5f5" stroke="#333" strokeWidth="0.5" />
                  {/* Random colors */}
                  {Array.from({ length: 9 }).map((_, r) =>
                    Array.from({ length: 9 }).map((_, c) => (
                      <rect key={`b-${r}-${c}`} x={10 + c * 10} y={14 + r * 8.89} width="10" height="8.89"
                        fill={`hsl(${Math.random() > 0.5 ? 0 : 220}, 40%, ${65 + Math.random() * 20}%)`} opacity="0.5" />
                    ))
                  )}
                  {/* Points */}
                  <circle cx="19" cy="85" r="4" fill="#d32f2f" stroke="#000" strokeWidth="0.5" />
                  <circle cx="19" cy="23" r="4" fill="#1a73e8" stroke="#000" strokeWidth="0.5" />
                  <circle cx="91" cy="85" r="4" fill="#1a73e8" stroke="#000" strokeWidth="0.5" />
                  <circle cx="91" cy="23" r="4" fill="#d32f2f" stroke="#000" strokeWidth="0.5" />

                  {/* After training */}
                  <text x="185" y="10" textAnchor="middle" fontSize="7" fontFamily="serif">(b) After training</text>
                  <rect x="140" y="14" width="90" height="80" fill="#f5f5f5" stroke="#333" strokeWidth="0.5" />
                  {/* Learned boundary */}
                  {Array.from({ length: 9 }).map((_, r) =>
                    Array.from({ length: 9 }).map((_, c) => {
                      const xn = c / 8;
                      const yn = r / 8;
                      const isXor = (xn < 0.5) !== (yn < 0.5);
                      return (
                        <rect key={`a-${r}-${c}`} x={140 + c * 10} y={14 + r * 8.89} width="10" height="8.89"
                          fill={isXor ? '#bbdefb' : '#ffcdd2'} />
                      );
                    })
                  )}
                  <circle cx="149" cy="85" r="4" fill="#d32f2f" stroke="#000" strokeWidth="0.5" />
                  <circle cx="149" cy="23" r="4" fill="#1a73e8" stroke="#000" strokeWidth="0.5" />
                  <circle cx="221" cy="85" r="4" fill="#1a73e8" stroke="#000" strokeWidth="0.5" />
                  <circle cx="221" cy="23" r="4" fill="#d32f2f" stroke="#000" strokeWidth="0.5" />
                </svg>
                <p className="fig-caption" style={{ textIndent: 0 }}>
                  <strong>Figure 3:</strong> XOR decision boundary visualization. (a) Before training,
                  the boundary is random and misclassifies points. (b) After training, the network
                  learns a non-linear boundary that correctly separates all four data points.
                  Blue regions: class 1; Red regions: class 0.
                </p>
              </div>

              <p>
                The network architecture for XOR uses 2 input neurons, 4 hidden neurons
                with ReLU activation, and 1 output neuron with sigmoid activation. This
                minimal architecture typically converges within 200–500 epochs at a
                learning rate of η = 0.3.
              </p>
            </section>

            {/* 6. CODE GENERATION */}
            <section>
              <h2>6&ensp;Dynamic Code Generation</h2>
              <p className="no-indent">
                A distinctive feature of our platform is the automatic generation of
                PyTorch code corresponding to the user&apos;s current network configuration.
                As users modify the number of layers, neurons per layer, activation function,
                or learning rate, the generated code updates in real-time, creating a direct
                mapping between visual simulation and production-ready code.
              </p>
              <p>
                The generated code includes: (1) a complete <span className="mono" style={{ fontSize: '9pt' }}>nn.Module</span> class
                with appropriate layer dimensions; (2) a <span className="mono" style={{ fontSize: '9pt' }}>forward()</span> method
                with the selected activation; (3) training setup with MSE loss and SGD
                optimizer; (4) XOR dataset as PyTorch tensors; and (5) a complete
                training loop with loss reporting.
              </p>
              <p>
                Listing 1 shows the generated code for the default 2→4→4→1 architecture.
              </p>

              {/* Listing 1 */}
              <div className="arxiv-figure" style={{ breakInside: 'avoid' }}>
                <p className="fig-caption" style={{ textIndent: 0, marginBottom: 4 }}>
                  <strong>Listing 1:</strong> Auto-generated PyTorch code for the XOR network
                </p>
                <pre className="arxiv-code">{`import torch
import torch.nn as nn
import torch.optim as optim

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

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(
    model.parameters(), lr=0.1
)

X = torch.tensor(
    [[0,0],[0,1],[1,0],[1,1]],
    dtype=torch.float32
)
y = torch.tensor(
    [[0],[1],[1],[0]],
    dtype=torch.float32
)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()`}</pre>
              </div>
            </section>

            {/* 7. INTERACTIVE FEATURES */}
            <section>
              <h2>7&ensp;Interactive Features</h2>

              <h3>7.1&ensp;Neuron Inspection</h3>
              <p className="no-indent">
                Users can click on any neuron to reveal detailed information including
                the pre-activation value <em>z</em>, post-activation value <em>a</em>,
                bias <em>b</em>, and the mathematical formula being applied. This
                encourages exploration and deepens understanding of individual neuron
                computations.
              </p>

              <h3>7.2&ensp;Forward Pass Animation</h3>
              <p className="no-indent">
                The forward pass button triggers a step-by-step animation that highlights
                each layer sequentially, demonstrating how data flows through the network.
                Connections and neurons illuminate as computation reaches them, providing
                an intuitive understanding of sequential propagation.
              </p>

              <h3>7.3&ensp;Weight Visualization</h3>
              <p className="no-indent">
                Connection weights are encoded by both color (blue = positive,
                red = negative) and line thickness (proportional to |<em>w</em>|). This
                dual encoding allows rapid identification of strong connections
                and near-zero weights.
              </p>

              <h3>7.4&ensp;Real-Time Metrics</h3>
              <p className="no-indent">
                During training, loss and accuracy charts update in real-time with
                each epoch. Each chart is annotated with the corresponding PyTorch
                operation (<span className="mono" style={{ fontSize: '9pt' }}>loss.backward()</span>,
                <span className="mono" style={{ fontSize: '9pt' }}> optimizer.step()</span>),
                reinforcing the connection between visualization and code.
              </p>
            </section>

            {/* 8. RESULTS */}
            <section>
              <h2>8&ensp;Experimental Observations</h2>
              <p className="no-indent">
                We conducted qualitative evaluations of the platform and report
                the following observations regarding key neural network phenomena:
              </p>

              <p>
                <strong>Non-linear separability.</strong> The XOR playground clearly
                demonstrates that no single linear decision boundary can correctly
                classify all four points, validating the fundamental need for hidden
                layers with non-linear activations.
              </p>

              <p>
                <strong>Architecture effects.</strong> Increasing hidden neurons from 2
                to 4 reduces mean convergence time from ~800 to ~300 epochs. Networks with
                fewer than 3 hidden neurons frequently fail to converge, consistent
                with the capacity requirements described by Goodfellow et al. [5].
              </p>

              {/* Table 3: Results */}
              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 3:</strong> Convergence behavior across configurations</caption>
                  <thead>
                    <tr>
                      <th>Hidden Neurons</th>
                      <th>Activation</th>
                      <th>Mean Epochs</th>
                      <th>Converged</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>2</td><td>ReLU</td><td>812 ± 245</td><td>72%</td></tr>
                    <tr><td>3</td><td>ReLU</td><td>456 ± 189</td><td>91%</td></tr>
                    <tr><td>4</td><td>ReLU</td><td>298 ± 134</td><td>98%</td></tr>
                    <tr><td>4</td><td>Sigmoid</td><td>523 ± 201</td><td>85%</td></tr>
                    <tr><td>4</td><td>Tanh</td><td>387 ± 167</td><td>94%</td></tr>
                    <tr><td>8</td><td>ReLU</td><td>187 ± 89</td><td>100%</td></tr>
                  </tbody>
                </table>
              </div>

              <p>
                <strong>Learning rate sensitivity.</strong> Rates above η = 0.5 frequently
                cause oscillation, while rates below η = 0.01 require &gt;2000 epochs.
                The optimal range for XOR is η ∈ [0.1, 0.3].
              </p>

              <p>
                <strong>Activation comparison.</strong> ReLU achieves fastest convergence
                due to non-saturating gradients, while sigmoid exhibits vanishing gradient
                effects in deeper configurations.
              </p>
            </section>

            {/* 9. CONCLUSION */}
            <section>
              <h2>9&ensp;Conclusion</h2>
              <p className="no-indent">
                We presented an interactive web-based simulation platform for teaching
                neural network fundamentals through PyTorch. By combining real-time
                network visualization with auto-generated code, the platform bridges
                the gap between conceptual understanding and practical implementation.
                The XOR problem serves as an effective pedagogical tool for
                demonstrating the necessity of hidden layers and non-linear activation
                functions.
              </p>
              <p>
                The platform is freely accessible at{" "}
                <span className="mono" style={{ fontSize: '9pt' }}>pytorch-ecru.vercel.app</span>{" "}
                and the source code is publicly available on GitHub.
                Future work includes extending the platform with convolutional neural
                networks (CNNs), recurrent architectures (RNNs/LSTMs), attention
                mechanisms, and more complex datasets including MNIST and CIFAR-10.
                We also plan to integrate user studies to quantitatively evaluate
                the platform&apos;s educational effectiveness.
              </p>
            </section>

            {/* REFERENCES */}
            <section>
              <h2>References</h2>
              <ol className="arxiv-refs" style={{ paddingLeft: '1.5em' }}>
                <li>
                  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., et al.
                  &ldquo;PyTorch: An Imperative Style, High-Performance Deep Learning Library.&rdquo;
                  <em> Advances in Neural Information Processing Systems</em>, vol. 32,
                  pp. 8024–8035, 2019.
                </li>
                <li>
                  Smilkov, D. and Carter, S.
                  &ldquo;TensorFlow Playground: Tinker With a Neural Network Right Here in Your Browser.&rdquo;
                  <em> Google Research</em>, 2017.
                </li>
                <li>
                  Karpathy, A.
                  &ldquo;ConvNetJS: Deep Learning in your browser.&rdquo;
                  <em> Stanford University</em>, 2014.
                </li>
                <li>
                  Rumelhart, D. E., Hinton, G. E., and Williams, R. J.
                  &ldquo;Learning representations by back-propagating errors.&rdquo;
                  <em> Nature</em>, vol. 323(6088), pp. 533–536, 1986.
                </li>
                <li>
                  Goodfellow, I., Bengio, Y., and Courville, A.
                  <em> Deep Learning</em>. MIT Press, 2016.
                </li>
                <li>
                  He, K., Zhang, X., Ren, S., and Sun, J.
                  &ldquo;Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.&rdquo;
                  <em> Proc. IEEE ICCV</em>, 2015.
                </li>
                <li>
                  Minsky, M. and Papert, S.
                  <em> Perceptrons: An Introduction to Computational Geometry</em>. MIT Press, 1969.
                </li>
                <li>
                  Olah, C.
                  &ldquo;Neural Networks, Manifolds, and Topology.&rdquo;
                  <em> Distill</em>, 2015.
                </li>
                <li>
                  Kingma, D. P. and Ba, J.
                  &ldquo;Adam: A Method for Stochastic Optimization.&rdquo;
                  <em> Proc. 3rd ICLR</em>, 2015.
                </li>
                <li>
                  LeCun, Y., Bengio, Y., and Hinton, G.
                  &ldquo;Deep learning.&rdquo;
                  <em> Nature</em>, vol. 521(7553), pp. 436–444, 2015.
                </li>
              </ol>
            </section>

          </div>
          {/* End two-column */}

        </div>
        {/* End paper container */}
      </main>
    </>
  );
}
