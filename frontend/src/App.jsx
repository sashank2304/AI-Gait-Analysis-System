import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Camera,
  CameraOff,
  HeartPulse,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
  Waves,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getHealth,
  getPrediction,
  resetSession,
  startCamera,
  stopCamera,
  videoFeedUrl,
} from "./services/api";

const initialPrediction = {
  cameraActive: false,
  collecting: true,
  framesCollected: 0,
  sequenceLength: 30,
  warmingUp: false,
  status: "OFFLINE",
  fallRisk: "WAITING",
  riskLevel: "waiting",
  anomalyScore: 0,
  gaitConfidence: 0,
  stabilityScore: 0,
  symmetryScore: 0,
  landmarksDetected: false,
  message: "Start the camera to begin gait analysis.",
};

const riskTheme = {
  low: {
    text: "text-emerald-200",
    bg: "bg-emerald-400/12",
    ring: "ring-emerald-300/30",
    accent: "#34d399",
  },
  medium: {
    text: "text-amber-200",
    bg: "bg-amber-400/12",
    ring: "ring-amber-300/30",
    accent: "#fbbf24",
  },
  high: {
    text: "text-rose-200",
    bg: "bg-rose-400/12",
    ring: "ring-rose-300/30",
    accent: "#fb7185",
  },
  waiting: {
    text: "text-cyan-100",
    bg: "bg-cyan-400/10",
    ring: "ring-cyan-300/25",
    accent: "#22d3ee",
  },
};

function formatScore(value) {
  return Number(value || 0).toFixed(5);
}

function App() {
  const [prediction, setPrediction] = useState(initialPrediction);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState(Date.now());

  const theme = riskTheme[prediction.riskLevel] || riskTheme.waiting;
  const riskGauge = useMemo(() => {
    const score = Math.min(100, Math.max(0, (prediction.anomalyScore / 0.09) * 100));
    return [{ name: "risk", value: Math.round(score), fill: theme.accent }];
  }, [prediction.anomalyScore, theme.accent]);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      getPrediction()
        .then((payload) => {
          setPrediction(payload);
          setError("");

          if (payload.cameraActive && payload.landmarksDetected) {
            setHistory((current) => {
              const next = [
                ...current,
                {
                  time: new Date(payload.timestamp * 1000).toLocaleTimeString([], {
                    minute: "2-digit",
                    second: "2-digit",
                  }),
                  anomalyScore: payload.anomalyScore,
                  stabilityScore: payload.stabilityScore,
                  symmetryScore: payload.symmetryScore,
                },
              ];
              return next.slice(-42);
            });
          }
        })
        .catch((err) => setError(err.message));
    }, 650);

    return () => window.clearInterval(timer);
  }, []);

  async function handleStart() {
    setIsBusy(true);
    setError("");

    try {
      await startCamera();
      setSessionId(Date.now());
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleStop() {
    setIsBusy(true);
    setError("");

    try {
      await stopCamera();
      setPrediction(initialPrediction);
      setHistory([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleReset() {
    setError("");
    await resetSession();
    setHistory([]);
  }

  return (
    <main className="grid-glow min-h-screen overflow-hidden px-4 py-5 sm:px-6 lg:px-8">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-5">
        <Header
          cameraActive={prediction.cameraActive}
          health={health}
          isBusy={isBusy}
          onStart={handleStart}
          onStop={handleStop}
          onReset={handleReset}
        />

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-lg border border-rose-300/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-100"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        <section className="grid gap-5 xl:grid-cols-[1.35fr_0.65fr]">
          <LiveVideoPanel
            cameraActive={prediction.cameraActive}
            collecting={prediction.collecting}
            warmingUp={prediction.warmingUp}
            framesCollected={prediction.framesCollected}
            sequenceLength={prediction.sequenceLength}
            landmarksDetected={prediction.landmarksDetected}
            sessionId={sessionId}
            theme={theme}
          />

          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-1">
            <StatusCard prediction={prediction} theme={theme} />
            <RiskGauge data={riskGauge} prediction={prediction} theme={theme} />
          </div>
        </section>

        <MetricGrid prediction={prediction} theme={theme} />

        <section className="grid gap-5 lg:grid-cols-[1fr_1fr_0.8fr]">
          <ChartPanel title="Live Anomaly Score" icon={Activity}>
            <ResponsiveContainer width="100%" height={230}>
              <AreaChart data={history}>
                <defs>
                  <linearGradient id="anomalyGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.55} />
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.14)" vertical={false} />
                <XAxis dataKey="time" hide />
                <YAxis width={42} domain={[0, 0.1]} tick={{ fill: "#8ea3b8", fontSize: 11 }} />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone"
                  dataKey="anomalyScore"
                  stroke="#22d3ee"
                  strokeWidth={3}
                  fill="url(#anomalyGradient)"
                  dot={false}
                  isAnimationActive
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartPanel>

          <ChartPanel title="Stability Trend" icon={Waves}>
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={history}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.14)" vertical={false} />
                <XAxis dataKey="time" hide />
                <YAxis width={34} domain={[0, 100]} tick={{ fill: "#8ea3b8", fontSize: 11 }} />
                <Tooltip content={<ChartTooltip />} />
                <Line
                  type="monotone"
                  dataKey="stabilityScore"
                  stroke="#34d399"
                  strokeWidth={3}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="symmetryScore"
                  stroke="#a78bfa"
                  strokeWidth={3}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>

          <InsightsPanel prediction={prediction} health={health} theme={theme} />
        </section>
      </section>
    </main>
  );
}

function Header({ cameraActive, health, isBusy, onStart, onStop, onReset }) {
  return (
    <header className="glass-panel rounded-lg px-4 py-4 sm:px-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">
            <BrainCircuit size={14} />
            AI Healthcare Analytics
          </div>
          <h1 className="text-2xl font-semibold tracking-normal text-white sm:text-3xl">
            Gait Analysis Dashboard
          </h1>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-300">
            Live BiLSTM reconstruction-error monitoring with MediaPipe pose landmarks.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <SystemPill active={cameraActive} label={cameraActive ? "Camera Live" : "Camera Off"} />
          <SystemPill active={Boolean(health?.modelReady ?? health?.ok)} label="Model Loaded" />

          <button
            type="button"
            onClick={onReset}
            className="inline-flex h-11 items-center gap-2 rounded-lg border border-slate-500/25 bg-slate-900/60 px-4 text-sm font-medium text-slate-100 transition hover:border-cyan-300/45 hover:text-cyan-100"
            title="Reset inference buffers"
          >
            <RefreshCcw size={17} />
            Reset
          </button>

          {cameraActive ? (
            <button
              type="button"
              onClick={onStop}
              disabled={isBusy}
              className="inline-flex h-11 items-center gap-2 rounded-lg bg-rose-400 px-4 text-sm font-semibold text-slate-950 transition hover:bg-rose-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <CameraOff size={18} />
              Stop Camera
            </button>
          ) : (
            <button
              type="button"
              onClick={onStart}
              disabled={isBusy}
              className="inline-flex h-11 items-center gap-2 rounded-lg bg-cyan-300 px-4 text-sm font-semibold text-slate-950 shadow-glow transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Camera size={18} />
              Start Camera
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

function SystemPill({ active, label }) {
  return (
    <span
      className={`inline-flex h-9 items-center gap-2 rounded-full border px-3 text-xs font-semibold ${
        active
          ? "border-emerald-300/35 bg-emerald-300/10 text-emerald-100"
          : "border-slate-500/25 bg-slate-900/50 text-slate-300"
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${active ? "bg-emerald-300" : "bg-slate-500"}`} />
      {label}
    </span>
  );
}

function LiveVideoPanel({
  cameraActive,
  collecting,
  warmingUp,
  framesCollected,
  sequenceLength,
  landmarksDetected,
  sessionId,
  theme,
}) {
  const progress = Math.min(100, (framesCollected / sequenceLength) * 100);

  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel overflow-hidden rounded-lg"
    >
      <div className="flex items-center justify-between border-b border-slate-500/15 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className={`rounded-lg p-2 ${theme.bg} ${theme.text}`}>
            <Camera size={19} />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">Live Pose Stream</h2>
            <p className="text-xs text-slate-400">
              {landmarksDetected ? "33 body landmarks detected" : "Awaiting full-body pose"}
            </p>
          </div>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ring-1 ${theme.bg} ${theme.text} ${theme.ring}`}>
          {cameraActive ? "LIVE" : "STANDBY"}
        </span>
      </div>

      <div className="video-frame relative bg-slate-950">
        {cameraActive ? (
          <img
            className="h-full w-full object-cover"
            src={videoFeedUrl(sessionId)}
            alt="Live gait analysis stream with pose landmarks"
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <div className="rounded-full border border-cyan-200/20 bg-cyan-300/10 p-5 text-cyan-100">
              <Camera size={36} />
            </div>
            <p className="max-w-md text-sm leading-6 text-slate-300">
              Camera stream is ready. Keep your full body in frame before starting analysis.
            </p>
          </div>
        )}

        {cameraActive && warmingUp && (
          <div className="absolute inset-x-4 bottom-4 rounded-lg border border-cyan-200/20 bg-slate-950/78 p-3 backdrop-blur">
            <div className="mb-2 flex items-center justify-between text-xs text-slate-200">
              <span>Live rolling window warm-up</span>
              <span>
                {framesCollected}/{sequenceLength}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-700/80">
              <div
                className="h-full rounded-full bg-cyan-300 transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </motion.section>
  );
}

function StatusCard({ prediction, theme }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-lg p-5"
    >
      <div className="mb-4 flex items-center justify-between">
        <div className={`rounded-lg p-2 ${theme.bg} ${theme.text}`}>
          <HeartPulse size={22} />
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-bold ring-1 ${theme.bg} ${theme.text} ${theme.ring}`}>
          {prediction.fallRisk}
        </span>
      </div>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Current Status</p>
      <h2 className={`mt-2 text-3xl font-semibold tracking-normal ${theme.text}`}>
        {prediction.status}
      </h2>
      <p className="mt-4 text-sm leading-6 text-slate-300">{prediction.message}</p>
    </motion.section>
  );
}

function RiskGauge({ data, prediction, theme }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-lg p-5"
    >
      <div className="mb-2 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Fall-Risk Indicator</p>
          <h3 className="mt-2 text-xl font-semibold text-white">{prediction.fallRisk}</h3>
        </div>
        <ShieldCheck className={theme.text} size={26} />
      </div>

      <div className="relative mx-auto h-48 max-w-xs">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="60%"
            innerRadius="72%"
            outerRadius="100%"
            barSize={16}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar background={{ fill: "rgba(148, 163, 184, 0.16)" }} dataKey="value" cornerRadius={999} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-x-0 bottom-8 text-center">
          <p className={`text-4xl font-semibold ${theme.text}`}>{data[0].value}</p>
          <p className="text-xs uppercase tracking-[0.16em] text-slate-400">Risk Load</p>
        </div>
      </div>
    </motion.section>
  );
}

function MetricGrid({ prediction, theme }) {
  const cards = [
    {
      label: "Anomaly Score",
      value: formatScore(prediction.anomalyScore),
      progress: Math.min(100, (prediction.anomalyScore / 0.09) * 100),
      icon: AlertTriangle,
      color: theme.accent,
    },
    {
      label: "Gait Confidence",
      value: `${prediction.gaitConfidence?.toFixed?.(1) ?? "0.0"}%`,
      progress: prediction.gaitConfidence,
      icon: BrainCircuit,
      color: "#22d3ee",
    },
    {
      label: "Stability Score",
      value: `${prediction.stabilityScore?.toFixed?.(1) ?? "0.0"}%`,
      progress: prediction.stabilityScore,
      icon: Activity,
      color: "#34d399",
    },
    {
      label: "Symmetry Score",
      value: `${prediction.symmetryScore?.toFixed?.(1) ?? "0.0"}%`,
      progress: prediction.symmetryScore,
      icon: Sparkles,
      color: "#a78bfa",
    },
  ];

  return (
    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map((card, index) => (
        <motion.article
          key={card.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.04 }}
          className="glass-panel rounded-lg p-4"
        >
          <div className="mb-4 flex items-center justify-between">
            <div className="rounded-lg bg-slate-900/80 p-2 text-slate-100">
              <card.icon size={19} />
            </div>
            <span className="text-xs text-slate-400">Live</span>
          </div>
          <p className="text-xs uppercase tracking-[0.16em] text-slate-400">{card.label}</p>
          <p className="mt-2 text-2xl font-semibold text-white">{card.value}</p>
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-700/70">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, card.progress || 0))}%`, background: card.color }}
            />
          </div>
        </motion.article>
      ))}
    </section>
  );
}

function ChartPanel({ title, icon: Icon, children }) {
  return (
    <section className="glass-panel rounded-lg p-4">
      <div className="mb-3 flex items-center gap-3">
        <div className="rounded-lg bg-cyan-300/10 p-2 text-cyan-100">
          <Icon size={18} />
        </div>
        <h2 className="text-sm font-semibold text-white">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) {
    return null;
  }

  return (
    <div className="rounded-lg border border-slate-500/25 bg-slate-950/95 px-3 py-2 text-xs text-slate-100 shadow-xl">
      <p className="mb-1 text-slate-400">{label}</p>
      {payload.map((item) => (
        <p key={item.dataKey} style={{ color: item.color }}>
          {item.name}: {Number(item.value).toFixed(item.dataKey === "anomalyScore" ? 5 : 1)}
        </p>
      ))}
    </div>
  );
}

function InsightsPanel({ prediction, health, theme }) {
  const checks = [
    ["Model", health?.ok ? "Loaded" : "Checking"],
    ["Input", `${prediction.sequenceLength || 30} x 99`],
    ["Pose", prediction.landmarksDetected ? "Detected" : "Searching"],
    ["Saved Threshold", health?.savedThreshold ? Number(health.savedThreshold).toFixed(5) : "Loaded"],
  ];

  return (
    <section className="glass-panel rounded-lg p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400">AI Health Insights</p>
          <h2 className="mt-2 text-lg font-semibold text-white">Inference Summary</h2>
        </div>
        <BrainCircuit className={theme.text} size={26} />
      </div>

      <p className="mb-4 text-sm leading-6 text-slate-300">{prediction.message}</p>

      <div className="space-y-3">
        {checks.map(([label, value]) => (
          <div key={label} className="flex items-center justify-between rounded-lg border border-slate-500/15 bg-slate-950/35 px-3 py-2">
            <span className="text-sm text-slate-400">{label}</span>
            <span className="text-sm font-semibold text-slate-100">{value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export default App;
