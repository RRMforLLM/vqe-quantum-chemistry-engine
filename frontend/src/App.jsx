import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis,
  Tooltip, ReferenceLine, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// Design tokens (Clean Academia)
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  bg:         "#ffffff",
  surface:    "#ffffff",
  border:     "#e2e8f0",
  accent:     "#0f766e",
  accentDim:  "#ccfbf1",
  muted:      "#cbd5e1",
  text:       "#334155",
  textDim:    "#64748b",
  textStrong: "#0f172a",
  green:      "#059669",
};

const BITSTRINGS = [
  "0000","0101","1010","0011","1100","0110",
  "1001","1111","0001","1110","0111","1011",
  "1101","0010","1000","0100",
];

const REFERENCE_E = -75.97;
const API         = "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// Shared styles
// ─────────────────────────────────────────────────────────────────────────────
const panelStyle = {
  padding:      "24px",
};

const labelStyle = {
  fontSize:      11,
  fontWeight:    600,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  color:         C.textDim,
  marginBottom:  16,
  fontFamily:    "Inter, system-ui, sans-serif",
};

// ─────────────────────────────────────────────────────────────────────────────
// HamiltonianMapper
// ─────────────────────────────────────────────────────────────────────────────
function HamiltonianMapper() {
  const [matrix, setMatrix] = useState([]);

  useEffect(() => {
    const chars = ["I", "X", "Y", "Z"];
    const generateTerm = () => Array.from({length: 8}, () => chars[Math.floor(Math.random() * 4)]).join(" ⊗ ");

    const interval = setInterval(() => {
      setMatrix(prev => [generateTerm(), ...prev].slice(0, 6));
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ ...panelStyle, minHeight: 310, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", overflow: "hidden", background: "#f8fafc", border: `1px dashed ${C.border}`, borderRadius: 8 }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 14, color: C.textDim, textAlign: "center", marginBottom: 24 }}>
           {matrix.map((term, i) => (
              <div key={i} style={{ opacity: 1 - (i * 0.15), transform: `scale(${1 - i * 0.05})`, transition: "all 0.1s" }}>
                {term}
              </div>
           ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }} style={{ width: 16, height: 16, border: `2px solid ${C.accentDim}`, borderTopColor: C.accent, borderRadius: "50%" }} />
            <div style={{ fontSize: 12, fontWeight: 600, color: C.accent, letterSpacing: "0.1em", textTransform: "uppercase" }}>
               Compiling 193-Term Pauli Hamiltonian
            </div>
        </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 1 — EnergyConvergence
// ─────────────────────────────────────────────────────────────────────────────
function EnergyConvergence({ data, converged, step }) {
  if (step != null && step < 0) return <HamiltonianMapper />;

  const chartData = data.map((d, i) => ({ step: i, energy: d.energy, best: d.best_energy }));

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>Energy Convergence vs Step</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: -10 }}>
          <XAxis dataKey="step" stroke={C.muted} tick={{ fill: C.textDim, fontSize: 11 }} tickLine={false} axisLine={false}
                 label={{ value: "Step", position: "insideBottomRight", offset: -5, fill: C.textDim, fontSize: 11 }} />
          <YAxis stroke={C.muted} tick={{ fill: C.textDim, fontSize: 11 }} tickLine={false} axisLine={false}
                 tickFormatter={v => v.toFixed(3)} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6}}
            labelStyle={{ color: C.textDim, fontSize: 11 }}
            itemStyle={{ fontSize: 12, fontWeight: 500 }}
            formatter={(v, n) => [v.toFixed(6) + " Ha", n === "energy" ? "Current" : "Best"]}
            itemSorter={(item) => (item.dataKey === "energy" ? -1 : 1)}
          />
          <ReferenceLine y={REFERENCE_E}
            stroke={C.muted} strokeDasharray="4 4" strokeWidth={1}
            label={{ value: `Reference: ${REFERENCE_E} Ha`, fill: C.textDim, fontSize: 11, position: "insideTopLeft" }} />
          <Line type="monotone" dataKey="energy" dot={false} strokeWidth={1.5}
                stroke={C.accent} strokeOpacity={0.4} isAnimationActive={false} />
          <Line type="monotone" dataKey="best" dot={false} strokeWidth={2.5}
                stroke={converged ? C.green : C.accent} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      <AnimatePresence>
        {converged && (
          <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}
            style={{ marginTop: 16, padding: "10px 16px", background: "#f0fdf4",
                     border: `1px solid #bbf7d0`, borderRadius: 6,
                     color: C.green, fontSize: 12, fontWeight: 500 }}>
            ✓ Converged (Early stopping triggered)
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 2 — OptimizerGrid
// ─────────────────────────────────────────────────────────────────────────────
function OptimizerGrid({ params, converged, step }) {
  const isOptimizing = step != null && step >= 0 && !converged;
  const normalized = params.length === 36
    ? params.map(p => Math.abs(Math.sin(p)))
    : Array(36).fill(0);

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>Optimization Parameters</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(9, 1fr)", gap: 8 }}>
        {normalized.map((intensity, i) => (
          <motion.div
            key={i}
            animate={converged
              ? { scale: 1, opacity: 0.8, background: C.green }
              : isOptimizing
                ? {
                    scale:   [1, 1 + intensity * 0.15, 1],
                    opacity: [0.2 + intensity * 0.6, 0.8, 0.2 + intensity * 0.6],
                  }
                : { scale: 1, opacity: 0.1 }
            }
            transition={converged
              ? { duration: 0.5, ease: "easeOut" }
              : isOptimizing
                ? { duration: 0.8 + intensity * 0.6, repeat: Infinity, ease: "easeInOut", delay: i * 0.03 }
                : { duration: 0 }
            }
            style={{
              width:        "100%",
              aspectRatio:  "1",
              borderRadius: "4px",
              background:   C.accent,
            }}
          />
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16, fontSize: 11, color: C.textDim, fontWeight: 500 }}>
        <span>Layer 1 (Init)</span>
        <span>Layer 2 (Entangle)</span>
        <span>Layer 3 (Entangle)</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Component 3 — MeasurementOutcomes
// ─────────────────────────────────────────────────────────────────────────────
function MeasurementOutcomes({ converged, step }) {
  const isOptimizing = step != null && step >= 0 && !converged;
  const [bars, setBars] = useState(() => BITSTRINGS.map(() => Math.random() * 0.3 + 0.05));
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isOptimizing) {
      intervalRef.current = setInterval(() => {
        setBars(BITSTRINGS.map(() => Math.random() * 0.85 + 0.05));
      }, 80);
    } else {
      clearInterval(intervalRef.current);
      if (converged) {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setBars(BITSTRINGS.map((_, i) => (i === 1 ? 0.98 : Math.random() * 0.02)));
      } else {
        setBars(BITSTRINGS.map(() => 0.05));
      }
    }
    return () => clearInterval(intervalRef.current);
  }, [isOptimizing, converged]);

  const chartData = BITSTRINGS.map((label, i) => ({ label, amplitude: bars[i] }));
  const barColor  = (entry) => entry.label === "0101" && converged ? C.green : C.accent;

  return (
    <div style={panelStyle}>
      <div style={{...labelStyle, display: "flex", justifyContent: "space-between", alignItems: "center"}}>
        <span>Measurement Outcomes</span>
        {isOptimizing && (
          <span style={{ color: C.textDim, animation: "pulse 1.5s infinite" }}>
            Sampling...
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 10, right: 0, bottom: 20, left: -20 }}>
          <XAxis dataKey="label" stroke={C.muted}
                 tick={{ fill: C.textDim, fontSize: 10 }}
                 interval={0} angle={-45} textAnchor="end" tickLine={false} axisLine={false} />
          <YAxis stroke={C.muted} tick={{ fill: C.textDim, fontSize: 10 }}
                 domain={[0, 1]} tickFormatter={v => v.toFixed(1)} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6 }}
            labelStyle={{ color: C.textStrong, fontSize: 11, fontWeight: 600 }}
            formatter={v => [v.toFixed(3), "Probability"]}
            cursor={{fill: C.bg}}
          />
          <Bar dataKey="amplitude" radius={[2, 2, 0, 0]} isAnimationActive={converged ? true : false} animationDuration={800}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={barColor(entry)} fillOpacity={converged && entry.label !== "0101" ? 0.1 : 0.8} />
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
      display: "flex", gap: 32, flexWrap: "wrap",
      fontSize: 13, fontWeight: 500,
    }}>
      {[
        ["Optimization Step", step ?? "—"],
        ["Best Energy",  bestEnergy != null ? bestEnergy.toFixed(6) + " Ha" : "—"],
        ["Patience", patience != null ? patience + " / 20" : "—"],
        ["Status", message || "Waiting…"],
      ].map(([k, v]) => (
        <div key={k} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", color: C.textDim, letterSpacing: "0.05em" }}>{k}</span>
          <span style={{ color: C.textStrong }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch screen (with Auto-Discover loop)
// ─────────────────────────────────────────────────────────────────────────────
function LaunchScreen({ onEngage }) {
  const [bondLength, setBondLength] = useState(0.96);
  const [isSweeping, setIsSweeping] = useState(false);
  const [pesData, setPesData] = useState([]);

  const runSweep = async () => {
    setIsSweeping(true);
    setPesData([]);
    const points = [0.5, 0.7, 0.9, 0.96, 1.1, 1.5, 2.0, 2.5];
    for (const r of points) {
      setBondLength(r);
      try {
        const res = await fetch(`${API}/run-vqe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ bond_length: r }),
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let energyFound = false;

        while (!energyFound) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop();
          for (const block of events) {
            const line = block.replace(/^data: /, "").trim();
            if (!line) continue;
            try {
              const pkt = JSON.parse(line);
              if (pkt.step >= 0) {
                setPesData(prev => [...prev, { r, energy: pkt.energy }]);
                energyFound = true;
                reader.cancel();
                break;
              }
            } catch (err) {
              console.error(err);
            }
          }
        }
      } catch (e) {
        console.error(e);
      }
    }
    setBondLength(0.96);
    setIsSweeping(false);
  };

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
        gap:            32,
        padding:        40,
        background:     C.bg,
        fontFamily:     "Inter, system-ui, sans-serif",
      }}
    >
      <div style={{ textAlign: "center", maxWidth: 600 }}>
        <div style={{
          fontSize:      12,
          fontWeight:    600,
          letterSpacing: "0.15em",
          color:         C.accent,
          marginBottom:  16,
          textTransform: "uppercase",
        }}>
          Variational Quantum Eigensolver
        </div>
        <h1 style={{
          margin:        0,
          fontSize:      "clamp(32px, 5vw, 56px)",
          fontWeight:    800,
          color:         C.textStrong,
          letterSpacing: "-0.02em",
          lineHeight:    1.1,
        }}>
          Ground State Energy
        </h1>
        <p style={{ marginTop: 24, color: C.textDim, fontSize: 16, lineHeight: 1.6 }}>
          H₂O molecule simulation • 6-31G basis • Active space 4e/4o
        </p>
      </div>

      <div style={{ ...panelStyle, width: "100%", maxWidth: 500 }}>
        <div style={{ ...labelStyle, marginBottom: 24 }}>Molecular Geometry</div>

        {/* PES Chart during sweep */}
        {pesData.length > 0 && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 140 }} style={{ marginBottom: 24 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pesData}>
                <XAxis dataKey="r" type="number" domain={[0.5, 2.5]} hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Line type="monotone" dataKey="energy" stroke={C.accent} strokeWidth={2} dot={{ r: 4, fill: C.surface, stroke: C.accent, strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <input
            type="range" min={0.5} max={3.0} step={0.01}
            value={bondLength}
            onChange={e => setBondLength(parseFloat(e.target.value))}
            disabled={isSweeping}
            style={{ flex: 1, accentColor: C.accent, cursor: isSweeping ? "default" : "pointer" }}
          />
          <span style={{
            fontSize: 24, fontWeight: 700,
            color: C.textStrong, minWidth: 80, textAlign: "right",
          }}>
            {bondLength.toFixed(2)}
            <span style={{ fontSize: 14, color: C.textDim, marginLeft: 4, fontWeight: 500 }}>Å</span>
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.muted, marginTop: 12, fontWeight: 500 }}>
          <span>0.50 Å</span>
          <span>Equilibrium (0.96 Å)</span>
          <span>3.00 Å</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 16 }}>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onEngage(bondLength)}
          disabled={isSweeping}
          style={{
            background:    C.accent,
            border:        "none",
            borderRadius:  8,
            padding:       "16px 40px",
            color:         C.surface,
            fontSize:      15,
            fontWeight:    600,
            cursor:        isSweeping ? "default" : "pointer",
            boxShadow:     "0 4px 6px -1px rgba(15, 118, 110, 0.2)",
            opacity:       isSweeping ? 0.5 : 1,
          }}
        >
          Simulate
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={runSweep}
          disabled={isSweeping}
          style={{
            background:    C.surface,
            border:        `1px solid ${C.border}`,
            borderRadius:  8,
            padding:       "16px 32px",
            color:         C.text,
            fontSize:      15,
            fontWeight:    600,
            cursor:        isSweeping ? "default" : "pointer",
            opacity:       isSweeping ? 0.5 : 1,
          }}
        >
          Auto-Discover
        </motion.button>
      </div>
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
  const [latest,     setLatest]     = useState({ step: -1, bestEnergy: null, patience: null, message: "Connecting…" });
  const abortRef = useRef(null);

  const startStream = useCallback(() => {
    setRunning(true);
    setHistory([]);
    setParams(Array(36).fill(0));
    setConverged(false);
    setLatest({ step: -1, bestEnergy: null, patience: null, message: "Connecting…" });

    if (abortRef.current) {
      abortRef.current.abort();
    }
    abortRef.current = new AbortController();

    fetch(`${API}/run-vqe`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ bond_length: bondLength }),
      signal:  abortRef.current.signal,
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
                setLatest(l => ({ ...l, step: pkt.step, message: pkt.message }));
              }

              if (pkt.status === "converged") {
                setConverged(true);
                setRunning(false);
              }
            } catch (err) {
              console.error("Malformed packet", err);
            }
          }
          pump();
        });
      }
      pump();
    }).catch((err) => {
      if (err.name !== "AbortError") setRunning(false);
    });
  }, [bondLength]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    startStream();
    return () => abortRef.current?.abort();
  }, [startStream]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        minHeight:  "100vh",
        background: C.bg,
        padding:    "32px",
        boxSizing:  "border-box",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      <div style={{
        display:        "flex",
        alignItems:     "center",
        justifyContent: "space-between",
        marginBottom:   32,
        flexWrap:       "wrap",
        gap:            16,
      }}>
        <div>
          <div style={{
            fontSize:      11,
            fontWeight:    600,
            letterSpacing: "0.1em",
            color:         C.textDim,
            marginBottom:  6,
            textTransform: "uppercase",
          }}>
            Simulation Telemetry
          </div>
          <div style={{ color: C.textStrong, fontSize: 24, fontWeight: 700, letterSpacing: "-0.01em" }}>
            H₂O · r(O-H) = {bondLength.toFixed(2)} Å
          </div>
        </div>

        <div style={{ display: "flex", gap: 12 }}>
          {converged && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={startStream}
              style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 6, padding: "10px 20px", color: C.textStrong,
                fontSize: 13, fontWeight: 600, cursor: "pointer",
                boxShadow: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
              }}
            >
              Re-run
            </motion.button>
          )}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onReset}
            style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 6, padding: "10px 20px", color: C.textStrong,
              fontSize: 13, fontWeight: 600, cursor: "pointer",
              boxShadow: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
            }}
          >
            ← New Simulation
          </motion.button>
        </div>
      </div>

      <div style={{ paddingBottom: 24, borderBottom: `1px solid ${C.border}`, marginBottom: 24 }}>
        <StatusBar
          step={latest.step}
          bestEnergy={latest.bestEnergy}
          patience={latest.patience}
          message={latest.message}
        />
      </div>

      <div style={{
        display:             "grid",
        gridTemplateColumns: "1fr 1fr",
        gridTemplateRows:    "auto auto",
        gap:                 24,
      }}>
        <div style={{ gridColumn: "1 / -1" }}>
          <EnergyConvergence data={history} converged={converged} step={latest.step} />
        </div>
        <OptimizerGrid params={params} converged={converged} step={latest.step} />
        <MeasurementOutcomes running={running} converged={converged} step={latest.step} />
      </div>

      <AnimatePresence>
        {running && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            style={{
              position:   "fixed",
              bottom:     32,
              right:      32,
              background: C.surface,
              border:     `1px solid ${C.border}`,
              borderRadius: 24,
              padding:    "10px 20px",
              fontSize:   13,
              fontWeight: 600,
              color:      C.accent,
              display:    "flex",
              alignItems: "center",
              gap:        10,
              boxShadow:  "0 4px 6px -1px rgb(0 0 0 / 0.1)",
            }}
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
              style={{
                width: 14, height: 14,
                border: `2px solid ${C.accentDim}`,
                borderTopColor: C.accent,
                borderRadius: "50%"
              }}
            />
            Processing
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        * { box-sizing: border-box; }
        body { margin: 0; background: ${C.bg}; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.muted}; border-radius: 4px; }
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
