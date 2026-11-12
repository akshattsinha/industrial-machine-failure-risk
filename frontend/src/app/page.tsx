
"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Prediction = {
  failure_probability: number;
  risk_level: string;
};

type ApiResponse = {
  status: string;
  predictions: {
    logistic_regression: Prediction;
    pca_logistic_regression: Prediction;
    random_forest: Prediction;
  };
  note?: string;
};

export default function Home() {
  const [machineType, setMachineType] = useState("M");
  const [airTemperature, setAirTemperature] = useState("298.1");
  const [processTemperature, setProcessTemperature] = useState("308.6");
  const [rpm, setRpm] = useState("1551");
  const [torque, setTorque] = useState("42.8");
  const [toolWear, setToolWear] = useState("108");

  const [result, setResult] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function predictFailure() {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          machine_type: machineType,
          air_temperature: Number(airTemperature),
          process_temperature: Number(processTemperature),
          rotational_speed: Number(rpm),
          torque: Number(torque),
          tool_wear: Number(toolWear),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to prediction API."
      );
    } finally {
      setLoading(false);
    }
  }

  function riskColor(level: string) {
    if (level === "HIGH") return "text-red-600 bg-red-50";
    if (level === "MEDIUM") return "text-amber-700 bg-amber-50";
    return "text-emerald-700 bg-emerald-50";
  }

  function PredictionCard({
    title,
    prediction,
  }: {
    title: string;
    prediction: Prediction;
  }) {
    const probability = prediction.failure_probability * 100;

    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-sm font-medium text-slate-500">{title}</p>

        <div className="mt-4 flex items-end justify-between">
          <span className="text-3xl font-semibold tracking-tight text-slate-900">
            {probability.toFixed(1)}%
          </span>

          <span
            className={`rounded-full px-3 py-1 text-xs font-semibold ${riskColor(
              prediction.risk_level
            )}`}
          >
            {prediction.risk_level} RISK
          </span>
        </div>

        <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-blue-600 transition-all"
            style={{ width: `${probability}%` }}
          />
        </div>

        <p className="mt-3 text-xs text-slate-400">
          Estimated failure probability
        </p>
      </div>
    );
  }

  function SensorInput({
    label,
    value,
    onChange,
    unit,
  }: {
    label: string;
    value: string;
    onChange: (value: string) => void;
    unit: string;
  }) {
    return (
      <label className="block">
        <span className="text-sm font-medium text-slate-700">
          {label}
        </span>

        <div className="mt-2 flex items-center rounded-xl border border-slate-200 bg-slate-50 px-3 focus-within:border-blue-500">
          <input
            type="number"
            step="any"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-transparent py-3 text-sm text-slate-900 outline-none"
            required
          />
          <span className="text-xs text-slate-400">{unit}</span>
        </div>
      </label>
    );
  }

  return (
    <main className="min-h-screen bg-[#f5f7fb] text-slate-900">

      {/* HEADER */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">

          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-lg font-bold text-white">
                M
              </div>

              <div>
                <h1 className="text-lg font-bold tracking-tight">
                  Machine Intelligence
                </h1>
                <p className="text-xs text-slate-500">
                  Predictive Maintenance Platform
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full bg-emerald-50 px-4 py-2 text-xs font-medium text-emerald-700">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Prediction dashboard
          </div>

        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-10">

        {/* INTRO */}
        <section className="mb-8">
          <p className="text-sm font-medium text-blue-600">
            INDUSTRIAL ANALYTICS / OVERVIEW
          </p>

          <h2 className="mt-2 text-3xl font-bold tracking-tight">
            Machine Health Monitoring
          </h2>

          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Analyse machine sensor measurements and estimate failure
            risk using trained machine learning models.
          </p>
        </section>

        {/* MAIN GRID */}
        <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">

          {/* INPUT PANEL */}
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

            <div className="mb-6">
              <h3 className="text-lg font-semibold">
                Machine Sensor Data
              </h3>
              <p className="mt-1 text-sm text-slate-500">
                Enter the current machine measurements.
              </p>
            </div>

            <div className="mb-5">
              <label className="text-sm font-medium text-slate-700">
                Machine Type
              </label>

              <select
                value={machineType}
                onChange={(e) => setMachineType(e.target.value)}
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-sm outline-none focus:border-blue-500"
              >
                <option value="L">Low (L)</option>
                <option value="M">Medium (M)</option>
                <option value="H">High (H)</option>
              </select>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">

              <SensorInput
                label="Air Temperature"
                value={airTemperature}
                onChange={setAirTemperature}
                unit="K"
              />

              <SensorInput
                label="Process Temperature"
                value={processTemperature}
                onChange={setProcessTemperature}
                unit="K"
              />

              <SensorInput
                label="Rotational Speed"
                value={rpm}
                onChange={setRpm}
                unit="RPM"
              />

              <SensorInput
                label="Torque"
                value={torque}
                onChange={setTorque}
                unit="Nm"
              />

              <SensorInput
                label="Tool Wear"
                value={toolWear}
                onChange={setToolWear}
                unit="min"
              />

            </div>

            <button
              onClick={predictFailure}
              disabled={loading}
              className="mt-8 w-full rounded-xl bg-blue-600 px-5 py-4 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Analysing machine..." : "Analyse Machine"}
            </button>

            {error && (
              <p className="mt-4 rounded-xl bg-red-50 p-3 text-sm text-red-700">
                {error}
              </p>
            )}

          </section>

          {/* RESULTS PANEL */}
          <section className="space-y-6">

            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold">
                Prediction Results
              </h3>

              <p className="mt-1 text-sm text-slate-500">
                Model-based machine failure estimates.
              </p>

              {!result && (
                <div className="mt-8 rounded-xl border border-dashed border-slate-300 py-16 text-center">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-xl text-blue-600">
                    ↗
                  </div>

                  <p className="mt-4 font-medium text-slate-700">
                    No analysis available
                  </p>

                  <p className="mt-2 text-sm text-slate-400">
                    Enter machine sensor values and run an analysis.
                  </p>
                </div>
              )}

              {result && (
                <div className="mt-6 space-y-4">

                  <PredictionCard
                    title="Random Forest"
                    prediction={result.predictions.random_forest}
                  />

                  <PredictionCard
                    title="Logistic Regression"
                    prediction={result.predictions.logistic_regression}
                  />

                  <PredictionCard
                    title="PCA + Logistic Regression"
                    prediction={result.predictions.pca_logistic_regression}
                  />

                </div>
              )}
            </div>

            <div className="rounded-2xl border border-blue-100 bg-blue-50/70 p-5">
              <h3 className="font-semibold text-blue-950">
                Analysis information
              </h3>

              <p className="mt-2 text-sm leading-6 text-blue-900/70">
                Predictions are generated from the trained AI4I
                predictive maintenance models. Risk thresholds are
                provisional and have not been calibrated for real
                industrial operating conditions.
              </p>
            </div>

          </section>
        </div>

        {/* FOOTER */}
        <footer className="mt-12 border-t border-slate-200 pt-6 text-xs text-slate-400">
          Industrial Machine Failure Risk Prediction System
          <span className="float-right">Machine Learning Project</span>
        </footer>

      </div>
    </main>
  );
}
