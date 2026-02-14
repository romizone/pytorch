"use client";
import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

interface TrainingData {
  epoch: number;
  loss: number;
  accuracy: number;
}

interface TrainingChartProps {
  data: TrainingData[];
}

export default function TrainingChart({ data }: TrainingChartProps) {
  const displayData = data.length > 200 ? data.filter((_, i) => i % Math.ceil(data.length / 200) === 0) : data;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="bg-gray-800/50 backdrop-blur rounded-xl border border-gray-700 p-4">
        <h3 className="text-sm font-semibold text-orange-400 mb-3">Loss Curve</h3>
        <div className="text-xs text-gray-400 mb-2 font-mono">criterion = nn.MSELoss()</div>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={displayData}>
            <defs>
              <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f97316" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#f97316" }}
            />
            <Area type="monotone" dataKey="loss" stroke="#f97316" fill="url(#lossGrad)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
        <div className="mt-2 p-2 bg-gray-900/50 rounded text-xs text-gray-400 font-mono">
          <span className="text-orange-400">loss</span> = criterion(output, target)
          <br />
          <span className="text-orange-400">loss</span>.backward() &nbsp;
          <span className="text-gray-500"># Compute gradients</span>
        </div>
      </div>

      <div className="bg-gray-800/50 backdrop-blur rounded-xl border border-gray-700 p-4">
        <h3 className="text-sm font-semibold text-green-400 mb-3">Accuracy</h3>
        <div className="text-xs text-gray-400 mb-2 font-mono">predictions = (output &gt; 0.5).float()</div>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={displayData}>
            <defs>
              <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6b7280" domain={[0, 1]} tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#22c55e" }}
              formatter={(value: number | undefined) => [`${((value ?? 0) * 100).toFixed(1)}%`, "Accuracy"]}
            />
            <Area type="monotone" dataKey="accuracy" stroke="#22c55e" fill="url(#accGrad)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
        <div className="mt-2 p-2 bg-gray-900/50 rounded text-xs text-gray-400 font-mono">
          <span className="text-green-400">accuracy</span> = (predictions == target).sum() / len(target)
        </div>
      </div>
    </div>
  );
}
