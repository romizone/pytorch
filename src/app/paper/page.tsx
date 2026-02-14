"use client";
import React from "react";
import Link from "next/link";

export default function PaperPage() {
  return (
    <main className="min-h-screen bg-white text-gray-900 print:bg-white">
      {/* Print Button */}
      <div className="fixed top-4 right-4 flex gap-2 print:hidden z-50">
        <Link
          href="/"
          className="px-4 py-2 bg-gray-800 text-white rounded-lg text-sm hover:bg-gray-700 transition-colors"
        >
          Back to Simulation
        </Link>
        <button
          onClick={() => window.print()}
          className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm hover:bg-orange-700 transition-colors"
        >
          Print / Save PDF
        </button>
      </div>

      <article className="max-w-4xl mx-auto px-8 py-12 font-serif leading-relaxed">
        {/* Title */}
        <header className="text-center mb-12 border-b-2 border-gray-200 pb-8">
          <h1 className="text-3xl font-bold mb-4 leading-tight">
            Interactive Neural Network Simulation for PyTorch Education:
            <br />
            A Web-Based Approach to Understanding Deep Learning Fundamentals
          </h1>

          <div className="mt-6 text-base">
            <p className="font-semibold">Romin Urismanto</p>
            <p className="text-gray-600 text-sm mt-1">
              Department of Computer Science
            </p>
            <p className="text-gray-500 text-sm mt-1">
              Email: rominurismanto@gmail.com
            </p>
          </div>

          <div className="mt-4 text-sm text-gray-500">
            February 2026
          </div>
        </header>

        {/* Abstract */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3 text-center">Abstract</h2>
          <div className="bg-gray-50 p-6 rounded-lg text-sm text-justify leading-relaxed">
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
          <p className="text-xs text-gray-500 mt-2">
            <strong>Keywords:</strong> Neural Networks, PyTorch, Interactive Simulation, Deep Learning Education,
            Web-Based Learning, Backpropagation Visualization, XOR Problem
          </p>
        </section>

        {/* 1. Introduction */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">1. Introduction</h2>
          <p className="text-justify mb-3">
            Deep learning has become a transformative technology across numerous domains, from computer
            vision and natural language processing to healthcare and autonomous systems. As demand for
            machine learning practitioners grows, effective educational tools become increasingly important.
            PyTorch, developed by Meta AI, has emerged as one of the leading frameworks for deep learning
            research and production, valued for its dynamic computational graph and Pythonic interface.
          </p>
          <p className="text-justify mb-3">
            Despite abundant textbooks and online courses, many learners struggle with the abstract
            mathematical concepts underlying neural networks. Traditional static diagrams fail to convey
            the dynamic nature of training processes such as forward propagation, gradient computation,
            and weight updates. This gap motivates the development of interactive simulation tools that
            allow learners to manipulate network parameters in real-time and observe immediate consequences.
          </p>
          <p className="text-justify mb-3">
            This paper presents an interactive web-based platform that combines neural network
            visualization with PyTorch code generation. The key contributions include:
          </p>
          <ul className="list-disc pl-8 mb-3 space-y-1">
            <li>A real-time neural network visualizer with configurable architecture and hyperparameters</li>
            <li>Live training simulation with backpropagation and gradient descent</li>
            <li>An XOR decision boundary playground demonstrating non-linear classification</li>
            <li>Auto-generated PyTorch code that reflects the user&apos;s network configuration</li>
            <li>Interactive exploration of core PyTorch concepts with mathematical foundations</li>
          </ul>
        </section>

        {/* 2. Related Work */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">2. Related Work</h2>
          <p className="text-justify mb-3">
            Several interactive tools have been developed for neural network education. TensorFlow
            Playground provides a browser-based visualization of simple neural networks for
            classification tasks. ConvNetJS by Andrej Karpathy offers JavaScript-based implementations
            of convolutional neural networks. Distill.pub has pioneered interactive articles that
            combine narrative with manipulable visualizations.
          </p>
          <p className="text-justify mb-3">
            Our work differentiates itself by explicitly mapping visual interactions to PyTorch code,
            providing learners with both conceptual understanding and practical implementation skills.
            Unlike previous tools that operate in isolation from production frameworks, our platform
            generates valid PyTorch code in real-time as users modify network architectures and
            hyperparameters.
          </p>
        </section>

        {/* 3. System Architecture */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">3. System Architecture</h2>

          <h3 className="text-lg font-semibold mb-2">3.1 Technology Stack</h3>
          <p className="text-justify mb-3">
            The platform is built using the following technology stack:
          </p>
          <table className="w-full border-collapse border border-gray-300 mb-4 text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-300 px-4 py-2 text-left">Component</th>
                <th className="border border-gray-300 px-4 py-2 text-left">Technology</th>
                <th className="border border-gray-300 px-4 py-2 text-left">Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 px-4 py-2">Framework</td>
                <td className="border border-gray-300 px-4 py-2">Next.js 16</td>
                <td className="border border-gray-300 px-4 py-2">Server-side rendering, routing</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-4 py-2">Language</td>
                <td className="border border-gray-300 px-4 py-2">TypeScript</td>
                <td className="border border-gray-300 px-4 py-2">Type safety, developer experience</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-4 py-2">Styling</td>
                <td className="border border-gray-300 px-4 py-2">Tailwind CSS</td>
                <td className="border border-gray-300 px-4 py-2">Utility-first CSS framework</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-4 py-2">Animations</td>
                <td className="border border-gray-300 px-4 py-2">Framer Motion</td>
                <td className="border border-gray-300 px-4 py-2">Smooth UI transitions</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-4 py-2">Charts</td>
                <td className="border border-gray-300 px-4 py-2">Recharts</td>
                <td className="border border-gray-300 px-4 py-2">Training metrics visualization</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="border border-gray-300 px-4 py-2">Visualization</td>
                <td className="border border-gray-300 px-4 py-2">SVG</td>
                <td className="border border-gray-300 px-4 py-2">Network graph rendering</td>
              </tr>
              <tr>
                <td className="border border-gray-300 px-4 py-2">Deployment</td>
                <td className="border border-gray-300 px-4 py-2">Vercel</td>
                <td className="border border-gray-300 px-4 py-2">Serverless hosting, CDN</td>
              </tr>
            </tbody>
          </table>

          <h3 className="text-lg font-semibold mb-2">3.2 Component Architecture</h3>
          <p className="text-justify mb-3">
            The application follows a modular component architecture with five primary modules:
          </p>
          <ol className="list-decimal pl-8 mb-3 space-y-2">
            <li>
              <strong>NeuralNetworkVisualizer:</strong> The core simulation component that renders
              an SVG-based network graph with interactive neurons, weighted connections, and
              forward pass animations. Users can dynamically modify the network architecture
              (2-6 layers, 1-8 neurons per hidden layer) and observe real-time weight changes
              during training.
            </li>
            <li>
              <strong>TrainingChart:</strong> Real-time visualization of training metrics using
              area charts. Displays loss curves and accuracy metrics with PyTorch code annotations
              explaining each mathematical operation.
            </li>
            <li>
              <strong>PyTorchConcepts:</strong> An interactive reference guide covering six
              fundamental PyTorch concepts: Tensors, Linear Layers, Activation Functions,
              Backpropagation, Optimizers, and Loss Functions. Each concept includes code
              examples, mathematical formulas, and animated visualizations.
            </li>
            <li>
              <strong>XORPlayground:</strong> A dedicated simulation for the XOR classification
              problem, featuring a real-time decision boundary heatmap that updates as the
              network trains. This demonstrates why non-linear activation functions are essential.
            </li>
            <li>
              <strong>TrainingPipeline:</strong> A step-by-step walkthrough of the complete
              PyTorch training workflow, from model definition through evaluation.
            </li>
          </ol>
        </section>

        {/* 4. Neural Network Engine */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">4. Neural Network Engine Implementation</h2>

          <h3 className="text-lg font-semibold mb-2">4.1 Network Representation</h3>
          <p className="text-justify mb-3">
            The neural network is represented as a collection of neurons and connections.
            Each neuron stores its position (for visualization), raw value (pre-activation),
            activated value (post-activation), and bias. Connections store source and destination
            indices along with weight values. Weights are initialized using He initialization:
          </p>
          <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm mb-4 text-center">
            w ~ N(0, &radic;(2/n<sub>in</sub>))
          </div>
          <p className="text-justify mb-3">
            where n<sub>in</sub> is the number of input neurons to the layer.
          </p>

          <h3 className="text-lg font-semibold mb-2">4.2 Forward Propagation</h3>
          <p className="text-justify mb-3">
            Forward propagation computes activations layer by layer. For each neuron j in layer l:
          </p>
          <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm mb-4 text-center space-y-1">
            <p>z<sub>j</sub><sup>(l)</sup> = &sum;<sub>i</sub> w<sub>ij</sub> &middot; a<sub>i</sub><sup>(l-1)</sup> + b<sub>j</sub><sup>(l)</sup></p>
            <p>a<sub>j</sub><sup>(l)</sup> = &sigma;(z<sub>j</sub><sup>(l)</sup>)</p>
          </div>
          <p className="text-justify mb-3">
            The platform supports three activation functions: ReLU (max(0, x)),
            Sigmoid (1/(1+e<sup>-x</sup>)), and Tanh. The output layer always uses sigmoid
            activation for the XOR binary classification task.
          </p>

          <h3 className="text-lg font-semibold mb-2">4.3 Backpropagation and Gradient Descent</h3>
          <p className="text-justify mb-3">
            Training uses stochastic gradient descent with backpropagation. The loss function
            is Mean Squared Error (MSE):
          </p>
          <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm mb-4 text-center">
            L = (1/n) &sum; (y&#x0302;<sub>i</sub> - y<sub>i</sub>)&sup2;
          </div>
          <p className="text-justify mb-3">
            Gradients are computed using the chain rule. For the output layer:
          </p>
          <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm mb-4 text-center">
            &delta;<sub>out</sub> = (y&#x0302; - y) &middot; &sigma;&prime;(z)
          </div>
          <p className="text-justify mb-3">
            For hidden layers, the error is propagated backward:
          </p>
          <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm mb-4 text-center">
            &delta;<sub>h</sub> = (&sum; w &middot; &delta;<sub>next</sub>) &middot; f&prime;(z)
          </div>
          <p className="text-justify mb-3">
            Weight updates follow the gradient descent rule:
          </p>
          <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm mb-4 text-center">
            w &larr; w - &eta; &middot; &delta; &middot; a<sub>prev</sub>
          </div>
          <p className="text-justify mb-3">
            where &eta; is the learning rate, configurable between 0.001 and 1.0 via
            a slider control.
          </p>
        </section>

        {/* 5. XOR Problem */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">5. The XOR Problem as a Teaching Tool</h2>
          <p className="text-justify mb-3">
            The XOR (exclusive or) function is a classic problem in neural network education because
            it cannot be solved by a single-layer perceptron. The XOR truth table produces outputs
            that are not linearly separable in the input space:
          </p>
          <table className="mx-auto border-collapse border border-gray-300 mb-4 text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-300 px-6 py-2">x₁</th>
                <th className="border border-gray-300 px-6 py-2">x₂</th>
                <th className="border border-gray-300 px-6 py-2">XOR Output</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="border border-gray-300 px-6 py-2 text-center">0</td><td className="border border-gray-300 px-6 py-2 text-center">0</td><td className="border border-gray-300 px-6 py-2 text-center">0</td></tr>
              <tr className="bg-gray-50"><td className="border border-gray-300 px-6 py-2 text-center">0</td><td className="border border-gray-300 px-6 py-2 text-center">1</td><td className="border border-gray-300 px-6 py-2 text-center">1</td></tr>
              <tr><td className="border border-gray-300 px-6 py-2 text-center">1</td><td className="border border-gray-300 px-6 py-2 text-center">0</td><td className="border border-gray-300 px-6 py-2 text-center">1</td></tr>
              <tr className="bg-gray-50"><td className="border border-gray-300 px-6 py-2 text-center">1</td><td className="border border-gray-300 px-6 py-2 text-center">1</td><td className="border border-gray-300 px-6 py-2 text-center">0</td></tr>
            </tbody>
          </table>
          <p className="text-justify mb-3">
            Our XOR Playground component visualizes the decision boundary as a 2D heatmap that
            updates in real-time during training. Blue regions indicate predictions close to 1,
            while red regions indicate predictions close to 0. Users can observe how the hidden
            layer creates a non-linear transformation of the input space, effectively folding
            the 2D plane to make the classes linearly separable in the hidden representation.
          </p>
          <p className="text-justify mb-3">
            The network architecture for XOR uses 2 input neurons, 4 hidden neurons with ReLU
            activation, and 1 output neuron with sigmoid activation. This minimal architecture
            is sufficient to learn the XOR function, typically converging within 200-500 epochs
            with a learning rate of 0.3.
          </p>
        </section>

        {/* 6. PyTorch Code Generation */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">6. Dynamic PyTorch Code Generation</h2>
          <p className="text-justify mb-3">
            A distinctive feature of our platform is the automatic generation of PyTorch code
            that corresponds to the user&apos;s current network configuration. As users modify the
            number of layers, neurons per layer, activation function, or learning rate, the
            generated PyTorch code updates in real-time. This creates a direct mapping between
            the visual simulation and production-ready code.
          </p>
          <p className="text-justify mb-3">
            The generated code includes:
          </p>
          <ul className="list-disc pl-8 mb-3 space-y-1">
            <li>A complete <code className="bg-gray-100 px-1 rounded text-sm">nn.Module</code> class definition with appropriate layer dimensions</li>
            <li>Forward method with the selected activation function</li>
            <li>Training setup with MSE loss and SGD optimizer</li>
            <li>XOR dataset definition as PyTorch tensors</li>
            <li>Complete training loop with loss reporting</li>
          </ul>
          <p className="text-justify mb-3">
            This approach helps learners transition from visual understanding to code
            implementation, a critical gap in many educational tools.
          </p>
        </section>

        {/* 7. Interactive Features */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">7. Interactive Features and Pedagogical Design</h2>

          <h3 className="text-lg font-semibold mb-2">7.1 Neuron Inspection</h3>
          <p className="text-justify mb-3">
            Users can click on any neuron in the network to reveal detailed information including
            the raw pre-activation value (z), post-activation value (a), bias (b), and the
            mathematical formula being applied. This feature encourages exploration and helps
            learners understand the role of each neuron in the network.
          </p>

          <h3 className="text-lg font-semibold mb-2">7.2 Forward Pass Animation</h3>
          <p className="text-justify mb-3">
            The &quot;Forward Pass&quot; button triggers a step-by-step animation that highlights
            each layer sequentially, demonstrating how data flows through the network from
            input to output. Connections and neurons illuminate as the computation reaches them,
            providing an intuitive understanding of the sequential nature of forward propagation.
          </p>

          <h3 className="text-lg font-semibold mb-2">7.3 Weight Visualization</h3>
          <p className="text-justify mb-3">
            Connection weights are encoded both by color (blue for positive, red for negative)
            and by thickness (proportional to absolute weight value). This dual encoding allows
            users to quickly identify strong positive connections, strong negative connections,
            and weak connections that contribute little to the computation.
          </p>

          <h3 className="text-lg font-semibold mb-2">7.4 Real-Time Training Metrics</h3>
          <p className="text-justify mb-3">
            During training, loss and accuracy charts update in real-time, providing immediate
            feedback on learning progress. Each chart is annotated with the corresponding
            PyTorch code, reinforcing the connection between visualization and implementation.
          </p>
        </section>

        {/* 8. Results */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">8. Demonstration Results</h2>
          <p className="text-justify mb-3">
            The platform successfully demonstrates several key neural network concepts:
          </p>
          <ol className="list-decimal pl-8 mb-3 space-y-2">
            <li>
              <strong>Non-linear separability:</strong> The XOR playground clearly shows that
              no single linear boundary can separate the four data points, validating the
              need for hidden layers and non-linear activations.
            </li>
            <li>
              <strong>Effect of architecture:</strong> Users can observe that adding more
              hidden neurons speeds convergence, while too few neurons prevent learning.
            </li>
            <li>
              <strong>Learning rate sensitivity:</strong> High learning rates cause oscillation
              and divergence, while very low rates lead to slow convergence. The slider
              allows real-time experimentation with this critical hyperparameter.
            </li>
            <li>
              <strong>Activation function comparison:</strong> Switching between ReLU, sigmoid,
              and tanh shows different convergence behaviors and gradient characteristics.
            </li>
            <li>
              <strong>Weight initialization effects:</strong> The reset button reinitializes
              weights randomly, demonstrating how different starting points affect training
              trajectories.
            </li>
          </ol>
        </section>

        {/* 9. Conclusion */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">9. Conclusion</h2>
          <p className="text-justify mb-3">
            This paper presented an interactive web-based simulation platform for teaching
            neural network fundamentals through PyTorch. By combining real-time visualization
            with auto-generated code, the platform bridges the gap between conceptual
            understanding and practical implementation. The XOR problem serves as an effective
            teaching tool for demonstrating the necessity of hidden layers and non-linear
            activation functions.
          </p>
          <p className="text-justify mb-3">
            The platform is freely accessible at{" "}
            <span className="text-blue-600">pytorch-ecru.vercel.app</span> and
            the source code is available on GitHub. Future work includes adding support for
            convolutional neural networks, recurrent architectures, and more complex datasets
            such as MNIST and CIFAR-10.
          </p>
        </section>

        {/* References */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">References</h2>
          <ol className="list-decimal pl-8 space-y-2 text-sm">
            <li>
              Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style,
              High-Performance Deep Learning Library. <em>Advances in Neural Information Processing
              Systems 32</em>, pp. 8024-8035.
            </li>
            <li>
              Smilkov, D. and Carter, S. (2017). TensorFlow Playground: Tinker With a Neural
              Network Right Here in Your Browser. <em>Google Research</em>.
            </li>
            <li>
              Karpathy, A. (2014). ConvNetJS: Deep Learning in your browser.
              <em>Stanford University</em>.
            </li>
            <li>
              Rumelhart, D.E., Hinton, G.E., and Williams, R.J. (1986). Learning representations
              by back-propagating errors. <em>Nature</em>, 323(6088), pp. 533-536.
            </li>
            <li>
              Goodfellow, I., Bengio, Y., and Courville, A. (2016). <em>Deep Learning</em>.
              MIT Press.
            </li>
            <li>
              He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving Deep into Rectifiers:
              Surpassing Human-Level Performance on ImageNet Classification.
              <em>Proceedings of the IEEE International Conference on Computer Vision</em>.
            </li>
            <li>
              Minsky, M. and Papert, S. (1969). <em>Perceptrons: An Introduction to Computational
              Geometry</em>. MIT Press.
            </li>
            <li>
              Olah, C. (2015). Neural Networks, Manifolds, and Topology. <em>Distill</em>.
            </li>
            <li>
              Kingma, D.P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization.
              <em>Proceedings of the 3rd International Conference on Learning Representations</em>.
            </li>
            <li>
              LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep learning. <em>Nature</em>,
              521(7553), pp. 436-444.
            </li>
          </ol>
        </section>

        {/* Appendix */}
        <section className="mb-8">
          <h2 className="text-xl font-bold mb-3">Appendix A: Complete PyTorch Implementation</h2>
          <div className="bg-gray-100 p-6 rounded-lg">
            <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-gray-800">{`import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class XORNet(nn.Module):
    """Neural network for XOR classification.

    Architecture: 2 -> 4 -> 4 -> 1
    Activation: ReLU (hidden), Sigmoid (output)
    """
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize model, loss, and optimizer
model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]],
                 dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]],
                 dtype=torch.float32)

# Training loop
for epoch in range(1000):
    # Forward pass
    output = model(X)
    loss = criterion(output, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        predictions = (output > 0.5).float()
        accuracy = (predictions == y).sum().item() / len(y)
        print(f"Epoch {epoch:4d} | "
              f"Loss: {loss.item():.6f} | "
              f"Accuracy: {accuracy:.2%}")

# Final evaluation
model.eval()
with torch.no_grad():
    final_output = model(X)
    print("\\nFinal Predictions:")
    for i in range(len(X)):
        print(f"  Input: {X[i].tolist()} -> "
              f"Predicted: {final_output[i].item():.4f} "
              f"(Rounded: {round(final_output[i].item())})")`}</pre>
          </div>
        </section>
      </article>

      {/* Print Styles */}
      <style jsx global>{`
        @media print {
          body {
            font-size: 11pt;
            color: black;
            background: white;
          }
          article {
            max-width: 100%;
            padding: 0;
          }
          @page {
            margin: 2cm;
          }
          section {
            page-break-inside: avoid;
          }
          pre {
            white-space: pre-wrap;
            word-wrap: break-word;
          }
        }
      `}</style>
    </main>
  );
}
