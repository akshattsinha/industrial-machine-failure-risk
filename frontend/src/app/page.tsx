
"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type MachineForm = {
  machine_type: string;
  air_temperature: string;
  process_temperature: string;
  rotational_speed: string;
  torque: string;
  tool_wear: string;
};

type PredictionResponse = {
  results?: Record<string, unknown>;
  detail?: string;
};

const initialForm: MachineForm = {
  machine_type: "M",
  air_temperature: "298.1",
  process_temperature: "308.6",
  rotational_speed: "1551",
  torque: "42.8",
  tool_wear: "108",
};

const modelSections = [
  {
    key: "bayes_rule",
    number: "01",
    title: "Bayes Rule",
    description: "Conditional probability and sensor-based failure risk analysis.",
  },
  {
    key: "logistic_regression",
    number: "02",
    title: "Logistic Regression",
    description: "Binary classification and estimated failure probability.",
  },
  {
    key: "pca",
    number: "03",
    title: "Principal Component Analysis",
    description: "Dimensionality reduction followed by failure classification.",
  },
  {
    key: "hidden_markov_model",
    number: "04",
    title: "Hidden Markov Model",
    description: "Inference of the most likely hidden operating state.",
  },
  {
    key: "random_forest",
    number: "05",
    title: "Random Forest",
    description: "Ensemble-based machine failure classification.",
  },
];

function formatLabel(key: string) {
  return key
    .replace(/_/g, " ")
    .replace(/\[.*?\]/g, "")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "Not available";

  if (typeof value === "boolean") {
    if (key.toLowerCase().includes("predicted_failure")) {
      return value ? "Failure predicted" : "No failure predicted";
    }
    return value ? "Yes" : "No";
  }

  if (typeof value === "number") {
    const percentageField =
      /probability|variance_retained|percentage|accuracy|precision|recall|f1|auc/i.test(
        key
      );

    if (percentageField && value >= 0 && value <= 1) {
      return `${(value * 100).toFixed(2)}%`;
    }

    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }

  if (typeof value === "string") return value;

  return "";
}

function ResultValue({
  label,
  value,
  depth = 0,
}: {
  label: string;
  value: unknown;
  depth?: number;
}) {
  if (value === null || value === undefined) {
    return (
      <div className="text-sm text-gray-400">
        {formatLabel(label)}: Not available
      </div>
    );
  }

  if (Array.isArray(value)) {
    return (
      <div className="space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {formatLabel(label)}
        </p>
        {value.map((item, index) => (
          <div
            key={index}
            className="rounded-lg bg-gray-50 p-3 text-sm text-gray-700"
          >
            {typeof item === "object" && item !== null ? (
              <ResultObject data={item as Record<string, unknown>} depth={depth + 1} />
            ) : (
              String(item)
            )}
          </div>
        ))}
      </div>
    );
  }

  if (typeof value === "object") {
    return (
      <div className="space-y-2">
        <p className="text-sm font-semibold text-gray-700">
          {formatLabel(label)}
        </p>
        <div className="rounded-lg bg-gray-50 p-3">
          <ResultObject
            data={value as Record<string, unknown>}
            depth={depth + 1}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start justify-between gap-4 border-b border-gray-100 py-3 last:border-0">
      <span className="text-sm text-gray-500">{formatLabel(label)}</span>
      <span className="max-w-[60%] text-right text-sm font-semibold text-gray-900">
        {formatValue(label, value)}
      </span>
    </div>
  );
}

function ResultObject({
  data,
  depth = 0,
}: {
  data: Record<string, unknown>;
  depth?: number;
}) {
  if (depth > 4) {
    return <p className="text-xs text-gray-500">Additional nested data</p>;
  }

  const entries = Object.entries(data);

  if (entries.length === 0) {
    return <p className="text-sm text-gray-400">No results available.</p>;
  }

  return (
    <div className="space-y-1">
      {entries.map(([key, value]) => (
        <ResultValue key={key} label={key} value={value} depth={depth} />
      ))}
    </div>
  );
}

function PredictionCard({
  number,
  title,
  description,
  prediction,
}: {
  number: string;
  title: string;
  description: string;
  prediction: unknown;
}) {
  const available =
    prediction !== null &&
    prediction !== undefined &&
    typeof prediction === "object";

  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex items-start gap-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-sm font-bold text-indigo-700">
          {number}
        </div>

        <div>
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          <p className="mt-1 text-sm text-gray-500">{description}</p>
        </div>
      </div>

      {available ? (
        <ResultObject data={prediction as Record<string, unknown>} />
      ) : (
        <p className="rounded-lg bg-amber-50 p-4 text-sm text-amber-800">
          Results are unavailable for this model. Check that its trained model
          artifact is loaded by the backend.
        </p>
      )}
    </section>
  );
}

export default function Home() {
  const [form, setForm] = useState<MachineForm>(initialForm);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateField(field: keyof MachineForm, value: string) {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  }

  async function analyseMachine(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResults(null);

    const payload = {
      machine_type: form.machine_type,
      air_temperature: Number(form.air_temperature),
      process_temperature: Number(form.process_temperature),
      rotational_speed: Number(form.rotational_speed),
      torque: Number(form.torque),
      tool_wear: Number(form.tool_wear),
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data: PredictionResponse = await response.json();

      if (!response.ok) {
        const errorMessage =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? data);

        throw new Error(
          `Backend error (${response.status}): ${errorMessage}`
        );
      }

      if (!data.results || typeof data.results !== "object") {
        throw new Error(
          "The backend response does not contain a valid results object. Check your FastAPI /predict endpoint."
        );
      }

      setResults(data.results);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction server."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f8fc] text-gray-900">
      <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8">
        {/* Header */}
        <header className="mb-10">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white px-3 py-1.5 text-xs font-medium text-indigo-700">
            <span className="h-2 w-2 rounded-full bg-green-500" />
            MACHINE LEARNING PROJECT
          </div>

          <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">
            Industrial Machine
            <span className="block text-indigo-600">
              Failure Risk Prediction
            </span>
          </h1>

          <p className="mt-4 max-w-2xl text-sm leading-6 text-gray-500 sm:text-base">
            Enter machine sensor readings to analyse failure risk using five
            machine learning and probabilistic techniques.
          </p>
        </header>

        {/* Input form */}
        <section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm sm:p-8">
          <div className="mb-7">
            <h2 className="text-xl font-semibold">Machine Sensor Parameters</h2>
            <p className="mt-1 text-sm text-gray-500">
              Enter the current operating values of the machine.
            </p>
          </div>

          <form onSubmit={analyseMachine}>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Machine Type
                </label>
                <select
                  value={form.machine_type}
                  onChange={(e) => updateField("machine_type", e.target.value)}
                  className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                >
                  <option value="L">L - Low</option>
                  <option value="M">M - Medium</option>
                  <option value="H">H - High</option>
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">
                  Air Temperature (K)
                </label>
                <input
                  required
                  type="number"
                  step="any"
                  value={form.air_temperature}
                  onChange={(e) =>
                    updateField("air_temperature", e.target.value)
                  }
                  className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">
                  Process Temperature (K)
                </label>
                <input
                  required
                  type="number"
                  step="any"
                  value={form.process_temperature}
                  onChange={(e) =>
                    updateField("process_temperature", e.target.value)
                  }
                  className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">
                  Rotational Speed (RPM)
                </label>
                <input
                  required
                  type="number"
                  step="any"
                  value={form.rotational_speed}
                  onChange={(e) =>
                    updateField("rotational_speed", e.target.value)
                  }
                  className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">
                  Torque (Nm)
                </label>
                <input
                  required
                  type="number"
                  step="any"
                  value={form.torque}
                  onChange={(e) => updateField("torque", e.target.value)}
                  className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">
                  Tool Wear (Minutes)
                </label>
                <input
                  required
                  type="number"
                  step="any"
                  value={form.tool_wear}
                  onChange={(e) => updateField("tool_wear", e.target.value)}
                  className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
                />
              </div>
            </div>

            {error && (
              <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                <strong>Analysis failed:</strong> {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="mt-7 flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-6 py-4 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
            >
              {loading ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Analysing Machine...
                </>
              ) : (
                "Analyse Machine →"
              )}
            </button>
          </form>
        </section>

        {/* Results */}
        {results && (
          <section className="mt-12">
            <div className="mb-7">
              <div className="text-xs font-semibold uppercase tracking-widest text-indigo-600">
                Analysis Complete
              </div>
              <h2 className="mt-2 text-2xl font-bold">
                Machine Analysis Results
              </h2>
              <p className="mt-2 text-sm text-gray-500">
                Outputs from all five selected academic topics.
              </p>
            </div>

            <div className="grid gap-5 lg:grid-cols-2">
              {modelSections.map((section) => (
                <PredictionCard
                  key={section.key}
                  number={section.number}
                  title={section.title}
                  description={section.description}
                  prediction={results[section.key]}
                />
              ))}
            </div>

            <div className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-5">
              <h3 className="font-semibold text-amber-900">
                Important interpretation notes
              </h3>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm leading-6 text-amber-800">
                <li>
                  The dataset contains synthetic machine operating data.
                </li>
                <li>
                  Model probabilities are estimates and should not be treated
                  as calibrated real-world failure probabilities.
                </li>
                <li>
                  The HMM output represents hidden-state inference from one
                  observation, not temporal degradation prediction.
                </li>
                <li>
                  Bayes Rule reports conditional probabilities for individual
                  sensor conditions; these values should not be added together.
                </li>
              </ul>
            </div>
          </section>
        )}

        {/* Initial state */}
        {!results && !loading && (
          <div className="mt-8 rounded-2xl border border-dashed border-gray-300 bg-white/60 px-6 py-10 text-center">
            <div className="text-3xl">⌁</div>
            <h3 className="mt-3 font-semibold text-gray-800">
              Ready to analyse your machine
            </h3>
            <p className="mt-2 text-sm text-gray-500">
              Enter the sensor parameters above and click Analyse Machine to
              view all five model outputs.
            </p>
          </div>
        )}

        <footer className="mt-12 border-t border-gray-200 pt-6 text-center text-xs text-gray-400">
          Industrial Machine Failure Risk Prediction · AI/ML Academic Project
        </footer>
      </div>
    </main>
  );
}
