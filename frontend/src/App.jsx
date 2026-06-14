import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  bg:       "#070b12",
  surface:  "#0d1520",
  border:   "#1a2740",
  blue:     "#38bdf8",
  blueDim:  "#0e4a6e",
  orange:   "#fb923c",
  orangeDim:"#5c2a0d",
  green:    "#34d399",
  muted:    "#4a6080",
  text:     "#cdd9e8",
  textDim:  "#6b8aaa",
};

const BITSTRINGS = [
  "|0000⟩","|0101⟩","|1010⟩","|0011⟩","|1100⟩","|0110⟩",
  "|1001⟩","|1111⟩","|0001⟩","|1110⟩","|0111⟩","|1011⟩",
  "|1101⟩","|0010⟩","|1000⟩","|0100⟩",
];

const REFERENCE_E = -75.97;
const API         = "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// Shared styles
// ─────────────────────────────────────────────────────────────────────────────
const panelStyle = {
  background:   C.surface,
  border:       `1px solid ${C.border}`,
  borderRadius: 12,
  padding:      "20px 24px",
};

const labelStyle = {
  fontSize:      10,
  letterSpacing: "0.18em",
  textTransform: "uppercase",
  color:         C.textDim,
  marginBottom:  10,
  fontFamily:    "'JetBrains Mono', 'Fira Code', monospace",
};

// ─────────────────────────────────────────────────────────────────────────────
// Component A — DescentTrace (live recharts line)
// ─────────────────────────────────────────────────────────────────────────────
function DescentTrace({ data, converged }) {
  const chartData = data.map((d, i) => ({ step: i, energy: d.energy, best: d.best_energy }));

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>SPSA Descent — Energy vs Step</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
          <XAxis dataKey="step" stroke={C.muted} tick={{ fill: C.textDim, fontSize: 10 }}
                 label={{ value: "Step", position: "insideBottomRight", offset: -4, fill: C.muted, fontSize: 10 }} />
          <YAxis stroke={C.muted} tick={{ fill: C.textDim, fontSize: 10 }}
                 tickFormatter={v => v.toFixed(3)} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8 }}
            labelStyle={{ color: C.textDim, fontSize: 10 }}
            itemStyle={{ fontSize: 11 }}
            formatter={(v, n) => [v.toFixed(6) + " Ha", n === "energy" ? "Current" : "Best"]}
          />
          <ReferenceLine y={REFERENCE_E}
            stroke={C.green} strokeDasharray="5 3" strokeWidth={1.5}
            label={{ value: `${REFERENCE_E} Ha`, fill: C.green, fontSize: 10, position: "insideTopLeft" }} />
          {/* raw trace — faint */}
          <Line type="monotone" dataKey="energy" dot={false} strokeWidth={1}
                stroke={C.blue} strokeOpacity={0.35} isAnimationActive={false} />
          {/* best envelope — vivid */}
          <Line type="monotone" dataKey="best" dot={false} strokeWidth={2}
                stroke={converged ? C.green : C.blue} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      {converged && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
          style={{ marginTop: 10, padding: "8px 14px", background: "#052015",
                   border: `1px solid ${C.green}`, borderRadius: 8,
                   color: C.green, fontSize: 11, fontFamily: "monospace", letterSpacing: "0.05em" }}>
          ✓ CONVERGED — early stopping triggered
        </motion.div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Component B — ParameterAnsatz (36-node HEA grid)
// ─────────────────────────────────────────────────────────────────────────────
function ParameterAnsatz({ params, converged }) {
  const normalized = params.length === 36
    ? params.map(p => Math.abs(Math.sin(p)))   // map angle → [0,1] pulse intensity
    : Array(36).fill(0);

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>HEA Parameters — 36-node ansatz</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(9, 1fr)", gap: 6 }}>
        {normalized.map((intensity, i) => (
          <motion.div
            key={i}
            animate={converged
              ? { scale: 1, opacity: 0.9, boxShadow: `0 0 8px ${C.green}66` }
              : {
                  scale:   [1, 1 + intensity * 0.25, 1],
                  opacity: [0.35 + intensity * 0.55, 0.9, 0.35 + intensity * 0.55],
                  boxShadow: [
                    `0 0 4px ${C.blueDim}`,
                    `0 0 ${6 + intensity * 14}px ${C.blue}`,
                    `0 0 4px ${C.blueDim}`,
                  ],
                }
            }
            transition={converged
              ? { duration: 0.6, ease: "easeOut" }
              : { duration: 0.4 + intensity * 0.5, repeat: Infinity, ease: "easeInOut", delay: i * 0.018 }
            }
            style={{
              width:        "100%",
              aspectRatio:  "1",
              borderRadius: "50%",
              background:   converged
                ? `radial-gradient(circle, ${C.green}88, ${C.green}22)`
                : `radial-gradient(circle, ${C.blue}${Math.round(55 + intensity * 150).toString(16).padStart(2,"0")}, ${C.blueDim}44)`,
              border:       `1px solid ${converged ? C.green : C.blue}44`,
              cursor:       "default",
            }}
          />
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12, fontSize: 10, color: C.textDim, fontFamily: "monospace" }}>
        <span>Init layer  (8 RY)</span>
        <span>Entangle L1  (14 CZ·RY)</span>
        <span>Entangle L2  (14 CZ·RY)</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Component C — QuantumState (bitstring bar chart + collapse animation)
// ─────────────────────────────────────────────────────────────────────────────
function QuantumState({ running, converged }) {
  const [bars, setBars] = useState(() => BITSTRINGS.map(() => Math.random() * 0.3 + 0.05));
  const intervalRef = useRef(null);

  useEffect(() => {
    if (running && !converged) {
      intervalRef.current = setInterval(() => {
        setBars(BITSTRINGS.map(() => Math.random() * 0.85 + 0.05));
      }, 120);
    } else {
      clearInterval(intervalRef.current);
      if (converged) {
        // collapse: spike on |0101⟩ (index 1) — the HF reference state
        setBars(BITSTRINGS.map((_, i) => (i === 1 ? 0.96 : Math.random() * 0.06)));
      }
    }
    return () => clearInterval(intervalRef.current);
  }, [running, converged]);

  const chartData = BITSTRINGS.map((label, i) => ({ label, amplitude: bars[i] }));
  const barColor  = (entry) => entry.label === "|0101⟩" && converged ? C.green : C.blue;

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>
        Quantum State — Shot Distribution
        {running && !converged && (
          <span style={{ marginLeft: 10, color: C.orange, animation: "pulse 1s infinite" }}>
            ● LIVE
          </span>
        )}
        {converged && (
          <span style={{ marginLeft: 10, color: C.green }}>✓ COLLAPSED</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 28, left: -10 }}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" stroke={C.muted}
                 tick={{ fill: C.textDim, fontSize: 8, fontFamily: "monospace" }}
                 interval={0} angle={-55} textAnchor="end" />
          <YAxis stroke={C.muted} tick={{ fill: C.textDim, fontSize: 9 }}
                 domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
          <Tooltip
            contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8 }}
            labelStyle={{ color: C.blue, fontSize: 10, fontFamily: "monospace" }}
            formatter={v => [v.toFixed(3), "Amplitude"]}
          />
          <Bar dataKey="amplitude" radius={[3, 3, 0, 0]} isAnimationActive={false}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={barColor(entry)} fillOpacity={converged && entry.label !== "|0101⟩" ? 0.25 : 0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Status bar
// ─────────────────────────────────────────────────────────────────────────────
function StatusBar({ step, bestEnergy, patience, message }) {
  return (
    <div style={{
      display: "flex", gap: 24, flexWrap: "wrap",
      fontFamily: "monospace", fontSize: 11,
    }}>
      {[
        ["STEP",    step ?? "—"],
        ["BEST E",  bestEnergy != null ? bestEnergy.toFixed(6) + " Ha" : "—"],
        ["PATIENCE",patience != null ? patience + " / 20" : "—"],
        ["MSG",     message || "Waiting…"],
      ].map(([k, v]) => (
        <div key={k} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ fontSize: 9, letterSpacing: "0.15em", color: C.textDim }}>{k}</span>
          <span style={{ color: C.text }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch screen
// ─────────────────────────────────────────────────────────────────────────────
function LaunchScreen({ onEngage }) {
  const [bondLength, setBondLength] = useState(0.96);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        minHeight:      "100vh",
        display:        "flex",
        flexDirection:  "column",
        alignItems:     "center",
        justifyContent: "center",
        gap:            40,
        padding:        40,
        background:     C.bg,
      }}
    >
      {/* Wordmark */}
      <div style={{ textAlign: "center" }}>
        <div style={{
          fontFamily:    "'JetBrains Mono', monospace",
          fontSize:      11,
          letterSpacing: "0.35em",
          color:         C.blue,
          marginBottom:  14,
        }}>
          VARIATIONAL QUANTUM EIGENSOLVER
        </div>
        <h1 style={{
          margin:        0,
          fontSize:      "clamp(36px, 6vw, 68px)",
          fontWeight:    700,
          color:         C.text,
          letterSpacing: "-0.02em",
          lineHeight:    1.1,
        }}>
          Ground State<br />
          <span style={{
            background:            `linear-gradient(90deg, ${C.blue}, ${C.orange})`,
            WebkitBackgroundClip:  "text",
            WebkitTextFillColor:   "transparent",
          }}>
            Energy Engine
          </span>
        </h1>
        <p style={{ marginTop: 16, color: C.textDim, fontSize: 14, maxWidth: 460 }}>
          H₂O · 6-31G basis · Active space 4e/4o · ibm_kingston Digital Twin
        </p>
      </div>

      {/* Slider */}
      <div style={{ ...panelStyle, width: "100%", maxWidth: 420 }}>
        <div style={{ ...labelStyle, marginBottom: 16 }}>O-H Bond Length</div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <input
            type="range" min={0.5} max={3.0} step={0.01}
            value={bondLength}
            onChange={e => setBondLength(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: C.blue, cursor: "pointer" }}
          />
          <span style={{
            fontFamily: "monospace", fontSize: 20, fontWeight: 600,
            color: C.blue, minWidth: 60, textAlign: "right",
          }}>
            {bondLength.toFixed(2)}
            <span style={{ fontSize: 12, color: C.textDim, marginLeft: 3 }}>Å</span>
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.muted, marginTop: 6, fontFamily: "monospace" }}>
          <span>0.50 Å</span>
          <span>Equilibrium ≈ 0.96 Å</span>
          <span>3.00 Å</span>
        </div>
      </div>

      {/* CTA */}
      <motion.button
        whileHover={{ scale: 1.04, boxShadow: `0 0 40px ${C.blue}55` }}
        whileTap={{ scale: 0.97 }}
        onClick={() => onEngage(bondLength)}
        style={{
          background:    `linear-gradient(135deg, ${C.blueDim}, #0a2a45)`,
          border:        `1.5px solid ${C.blue}`,
          borderRadius:  10,
          padding:       "16px 52px",
          color:         C.blue,
          fontSize:      15,
          fontFamily:    "'JetBrains Mono', monospace",
          fontWeight:    600,
          letterSpacing: "0.12em",
          cursor:        "pointer",
          transition:    "box-shadow 0.2s",
        }}
      >
        ENGAGE QPU
      </motion.button>
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard
// ─────────────────────────────────────────────────────────────────────────────
function Dashboard({ bondLength, onReset }) {
  const [history,    setHistory]    = useState([]);
  const [params,     setParams]     = useState(Array(36).fill(0));
  const [converged,  setConverged]  = useState(false);
  const [running,    setRunning]    = useState(false);
  const [latest,     setLatest]     = useState({ step: null, bestEnergy: null, patience: null, message: "Connecting…" });
  const esRef = useRef(null);

  const startStream = useCallback(() => {
    setRunning(true);

    esRef.current?.close();

    fetch(`${API}/run-vqe`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ bond_length: bondLength }),
    }).then(res => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      function pump() {
        reader.read().then(({ done, value }) => {
          if (done) { setRunning(false); return; }
          buffer += decoder.decode(value, { stream: true });

          // SSE lines: "data: {...}\n\n"
          const events = buffer.split("\n\n");
          buffer = events.pop();                  // keep incomplete tail

          for (const block of events) {
            const line = block.replace(/^data: /, "").trim();
            if (!line) continue;
            try {
              const pkt = JSON.parse(line);

              if (pkt.step >= 0) {
                setHistory(h => [...h, { energy: pkt.energy, best_energy: pkt.best_energy }]);
                setParams(pkt.params ?? Array(36).fill(0));
                setLatest({ step: pkt.step, bestEnergy: pkt.best_energy, patience: pkt.patience, message: pkt.message });
              } else {
                setLatest(l => ({ ...l, message: pkt.message }));
              }

              if (pkt.status === "converged") {
                setConverged(true);
                setRunning(false);
              }
            } catch (_) { /* malformed packet */ }
          }
          pump();
        });
      }
      pump();
    }).catch(() => setRunning(false));
  }, [bondLength]);

  useEffect(() => {
    startStream();
    return () => esRef.current?.close();
  }, [startStream]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        minHeight:  "100vh",
        background: C.bg,
        padding:    "24px 28px",
        boxSizing:  "border-box",
      }}
    >
      {/* Header */}
      <div style={{
        display:        "flex",
        alignItems:     "center",
        justifyContent: "space-between",
        marginBottom:   20,
        flexWrap:       "wrap",
        gap:            12,
      }}>
        <div>
          <div style={{
            fontFamily:    "monospace",
            fontSize:      10,
            letterSpacing: "0.3em",
            color:         C.blue,
            marginBottom:  4,
          }}>
            VQE TELEMETRY — ibm_kingston Digital Twin
          </div>
          <div style={{ color: C.text, fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em" }}>
            H₂O · r(O-H) = {bondLength.toFixed(2)} Å
            {converged && (
              <motion.span
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                style={{ marginLeft: 14, fontSize: 13, color: C.green, fontWeight: 400 }}
              >
                Ground state locked
              </motion.span>
            )}
          </div>
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          {converged && (
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={startStream}
              style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 8, padding: "8px 18px", color: C.textDim,
                fontSize: 11, fontFamily: "monospace", cursor: "pointer",
              }}
            >
              RE-RUN
            </motion.button>
          )}
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={onReset}
            style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: "8px 18px", color: C.textDim,
              fontSize: 11, fontFamily: "monospace", cursor: "pointer",
            }}
          >
            ← NEW RUN
          </motion.button>
        </div>
      </div>

      {/* Status bar */}
      <div style={{ ...panelStyle, marginBottom: 16 }}>
        <StatusBar
          step={latest.step}
          bestEnergy={latest.bestEnergy}
          patience={latest.patience}
          message={latest.message}
        />
      </div>

      {/* Main grid */}
      <div style={{
        display:             "grid",
        gridTemplateColumns: "1fr 1fr",
        gridTemplateRows:    "auto auto",
        gap:                 16,
      }}>
        {/* A — Descent trace (full width) */}
        <div style={{ gridColumn: "1 / -1" }}>
          <DescentTrace data={history} converged={converged} />
        </div>

        {/* B — Parameter grid */}
        <ParameterAnsatz params={params} converged={converged} />

        {/* C — Quantum state */}
        <QuantumState running={running} converged={converged} />
      </div>

      {/* Running indicator */}
      <AnimatePresence>
        {running && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position:   "fixed",
              bottom:     20,
              right:      24,
              background: C.surface,
              border:     `1px solid ${C.orange}`,
              borderRadius: 8,
              padding:    "8px 16px",
              fontFamily: "monospace",
              fontSize:   11,
              color:      C.orange,
              display:    "flex",
              alignItems: "center",
              gap:        8,
            }}
          >
            <motion.span
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ repeat: Infinity, duration: 1 }}
            >●</motion.span>
            QPU SAMPLING
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
      `}</style>
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Root
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [bondLength, setBondLength] = useState(null);

  return (
    <AnimatePresence mode="wait">
      {bondLength == null
        ? <LaunchScreen key="launch" onEngage={bl => setBondLength(bl)} />
        : <Dashboard    key="dash"   bondLength={bondLength} onReset={() => setBondLength(null)} />
      }
    </AnimatePresence>
  );
}
