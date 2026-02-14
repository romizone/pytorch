"use client";
import React, { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

const NeuralNetworkVisualizer = dynamic(() => import("@/components/NeuralNetworkVisualizer"), { ssr: false });
const TrainingChart = dynamic(() => import("@/components/TrainingChart"), { ssr: false });
const PyTorchConcepts = dynamic(() => import("@/components/PyTorchConcepts"), { ssr: false });
const XORPlayground = dynamic(() => import("@/components/XORPlayground"), { ssr: false });
const TrainingPipeline = dynamic(() => import("@/components/TrainingPipeline"), { ssr: false });

interface TrainingData {
  epoch: number;
  loss: number;
  accuracy: number;
}

export default function Home() {
  const [trainingData, setTrainingData] = useState<TrainingData[]>([]);
  const [activeSection, setActiveSection] = useState("simulation");

  const handleTrainingData = useCallback((data: TrainingData) => {
    setTrainingData((prev) => [...prev.slice(-500), data]);
  }, []);

  const sections = [
    { id: "simulation", label: "Live Simulation", icon: "⚡" },
    { id: "concepts", label: "PyTorch Concepts", icon: "📚" },
    { id: "xor", label: "XOR Playground", icon: "🎯" },
    { id: "pipeline", label: "Training Pipeline", icon: "🔧" },
  ];

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-indigo-950">
      {/* Hero Header */}
      <header className="relative overflow-hidden border-b border-gray-800">
        <div className="absolute inset-0 bg-gradient-to-r from-orange-600/10 via-transparent to-blue-600/10" />
        <div className="max-w-7xl mx-auto px-4 py-8 relative">
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold text-white">
                  PyTorch Neural Network
                  <span className="bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent"> Simulation</span>
                </h1>
                <p className="text-sm text-gray-400">Interactive Machine Learning Education Platform</p>
              </div>
            </div>
            <p className="text-gray-400 text-sm max-w-2xl mt-3">
              Explore how neural networks work through interactive simulations. Build, train, and visualize networks
              in real-time while learning PyTorch fundamentals with auto-generated code.
            </p>
          </motion.div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-gray-900/80 backdrop-blur-xl border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 py-2 overflow-x-auto">
            {sections.map((section) => (
              <button
                key={section.id}
                onClick={() => {
                  setActiveSection(section.id);
                  document.getElementById(section.id)?.scrollIntoView({ behavior: "smooth" });
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                  activeSection === section.id
                    ? "bg-orange-600 text-white shadow-lg shadow-orange-600/20"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                <span>{section.icon}</span>
                {section.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-12">
        {/* Section 1: Live Simulation */}
        <section id="simulation">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
            <div className="flex items-center gap-2 mb-6">
              <span className="text-xl">⚡</span>
              <h2 className="text-xl font-bold text-white">Live Neural Network Simulation</h2>
            </div>
            <div className="mb-2 p-3 bg-blue-600/10 border border-blue-500/20 rounded-lg">
              <p className="text-xs text-blue-300">
                <strong>How to use:</strong> Adjust the architecture (add/remove layers &amp; neurons), set hyperparameters,
                then click &quot;Train&quot; to watch the network learn the XOR function. Click any neuron to inspect its values.
                The PyTorch code below auto-updates to match your network configuration.
              </p>
            </div>
            <NeuralNetworkVisualizer onTrainingData={handleTrainingData} />

            {trainingData.length > 0 && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="mt-6">
                <TrainingChart data={trainingData} />
              </motion.div>
            )}
          </motion.div>
        </section>

        {/* Section 2: PyTorch Concepts */}
        <section id="concepts">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
            <div className="flex items-center gap-2 mb-6">
              <span className="text-xl">📚</span>
              <h2 className="text-xl font-bold text-white">Core PyTorch Concepts</h2>
            </div>
            <PyTorchConcepts />
          </motion.div>
        </section>

        {/* Section 3: XOR Playground */}
        <section id="xor">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
            <div className="flex items-center gap-2 mb-6">
              <span className="text-xl">🎯</span>
              <h2 className="text-xl font-bold text-white">XOR Decision Boundary Playground</h2>
            </div>
            <XORPlayground />
          </motion.div>
        </section>

        {/* Section 4: Training Pipeline */}
        <section id="pipeline">
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
            <div className="flex items-center gap-2 mb-6">
              <span className="text-xl">🔧</span>
              <h2 className="text-xl font-bold text-white">PyTorch Training Pipeline</h2>
            </div>
            <TrainingPipeline />
          </motion.div>
        </section>

        {/* Footer */}
        <footer className="border-t border-gray-800 pt-8 pb-4">
          <div className="text-center">
            <p className="text-gray-500 text-sm">
              PyTorch Neural Network Simulation &mdash; Interactive ML Education Platform
            </p>
            <p className="text-gray-600 text-xs mt-2">
              Built with Next.js, TypeScript, Tailwind CSS, and Framer Motion
            </p>
            <p className="text-gray-600 text-xs mt-1">
              Romin Urismanto &copy; {new Date().getFullYear()}
            </p>
          </div>
        </footer>
      </div>
    </main>
  );
}
