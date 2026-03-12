import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import axios from "axios";
import "./App.css";

// ─── Custom Animated SVG Bar Chart ───────────────────────────────────────────
function ConfidenceChart({ probabilities, prediction }) {
  const [animated, setAnimated] = useState(Array(10).fill(0));
  const [hovered, setHovered] = useState(null);
  const rafRef = useRef(null);
  const prevRef = useRef(Array(10).fill(0));

  useEffect(() => {
    const target = probabilities.map((v) => v / 100);
    const STIFFNESS = 0.12;

    const tick = () => {
      prevRef.current = prevRef.current.map((cur, i) => {
        const diff = target[i] - cur;
        if (Math.abs(diff) < 0.0005) return target[i];
        return cur + diff * STIFFNESS;
      });
      setAnimated([...prevRef.current]);
      const done = prevRef.current.every((v, i) => Math.abs(v - target[i]) < 0.001);
      if (!done) rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [probabilities]);

  const W = 560;
  const H = 320;
  const PAD_L = 42;
  const PAD_R = 16;
  const PAD_T = 20;
  const PAD_B = 38;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;
  const barW = (chartW / 10) * 0.58;
  const gap = chartW / 10;

  const gridLines = [0, 25, 50, 75, 100];

  return (
    <div className="chart-svg-wrap" style={{ flex: 1 }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ display: "block", width: "100%", height: "100%", overflow: "visible" }}
      >
        <defs>
          <linearGradient id="bar-grad-blue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#7ec8ff" stopOpacity="1" />
            <stop offset="100%" stopColor="#2166c4" stopOpacity="0.7" />
          </linearGradient>
          <linearGradient id="bar-grad-green" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#80ffca" stopOpacity="1" />
            <stop offset="100%" stopColor="#00b864" stopOpacity="0.7" />
          </linearGradient>
          <linearGradient id="shimmer" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="white" stopOpacity="0">
              <animate attributeName="offset" values="-0.8;1.2" dur="1.6s" repeatCount="indefinite" />
            </stop>
            <stop offset="50%" stopColor="white" stopOpacity="0.22">
              <animate attributeName="offset" values="-0.3;1.7" dur="1.6s" repeatCount="indefinite" />
            </stop>
            <stop offset="100%" stopColor="white" stopOpacity="0">
              <animate attributeName="offset" values="0.2;2.2" dur="1.6s" repeatCount="indefinite" />
            </stop>
          </linearGradient>
          <filter id="bar-glow" x="-40%" y="-20%" width="180%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="bar-glow-strong" x="-60%" y="-30%" width="220%" height="160%">
            <feGaussianBlur stdDeviation="7" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <style>{`
            @keyframes bar-pop {
              0%   { transform: scaleY(0.7); opacity: 0.4; }
              60%  { transform: scaleY(1.04); }
              100% { transform: scaleY(1); opacity: 1; }
            }
          `}</style>
        </defs>

        {gridLines.map((pct) => {
          const y = PAD_T + chartH - (pct / 100) * chartH;
          return (
            <g key={pct}>
              <line
                x1={PAD_L}
                y1={y}
                x2={W - PAD_R}
                y2={y}
                stroke={pct === 0 ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.06)"}
                strokeWidth={pct === 0 ? 1 : 0.5}
              />
              <text
                x={PAD_L - 5}
                y={y + 4}
                textAnchor="end"
                fontSize="9"
                fill="rgba(255,255,255,0.35)"
                fontFamily="'Courier New', monospace"
              >
                {pct}%
              </text>
            </g>
          );
        })}

        {animated.map((val, i) => {
          const isWinner = i === prediction;
          const isHov = hovered === i;
          const x = PAD_L + i * gap + (gap - barW) / 2;
          const barH = Math.max(0, val * chartH);
          const y = PAD_T + chartH - barH;
          const pct = (val * 100).toFixed(1);

          return (
            <g
              key={i}
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              {isWinner && barH > 2 && (
                <rect
                  x={x - 2}
                  y={y - 2}
                  width={barW + 4}
                  height={barH + 4}
                  rx="5"
                  fill="url(#bar-grad-green)"
                  opacity="0.25"
                  filter="url(#bar-glow-strong)"
                />
              )}

              {barH > 0.5 && (
                <rect
                  x={x}
                  y={y}
                  width={barW}
                  height={barH}
                  rx="5"
                  ry="5"
                  fill={isWinner ? "url(#bar-grad-green)" : "url(#bar-grad-blue)"}
                  opacity={isHov ? 1 : isWinner ? 0.92 : 0.78}
                  filter={isWinner ? "url(#bar-glow)" : isHov ? "url(#bar-glow)" : undefined}
                />
              )}

              {isWinner && barH > 2 && (
                <rect x={x} y={y} width={barW} height={barH} rx="5" fill="url(#shimmer)" opacity="0.6" />
              )}

              {barH > 3 && (
                <rect
                  x={x + 2}
                  y={y}
                  width={barW - 4}
                  height="2.5"
                  rx="1.5"
                  fill={isWinner ? "rgba(180,255,220,0.9)" : "rgba(150,210,255,0.7)"}
                />
              )}

              {(isHov || isWinner) && barH > 0 && (
                <text
                  x={x + barW / 2}
                  y={y - 6}
                  textAnchor="middle"
                  fontSize="10"
                  fontWeight="700"
                  fill={isWinner ? "#00ff8c" : "#7ec8ff"}
                  fontFamily="'Courier New', monospace"
                >
                  {pct}%
                </text>
              )}

              <text
                x={x + barW / 2}
                y={H - PAD_B + 14}
                textAnchor="middle"
                fontSize={isWinner ? "13" : "11"}
                fontWeight={isWinner ? "700" : "400"}
                fill={isWinner ? "#00ff8c" : isHov ? "#7ec8ff" : "rgba(255,255,255,0.55)"}
                fontFamily="'Courier New', monospace"
              >
                {i}
              </text>
            </g>
          );
        })}

        {hovered !== null &&
          (() => {
            const val = animated[hovered];
            if (val < 0.005) return null;
            const x = PAD_L + hovered * gap + gap / 2;
            const y = PAD_T + chartH - val * chartH;
            return (
              <circle
                cx={x}
                cy={y}
                r="4"
                fill={hovered === prediction ? "#00ff8c" : "#7ec8ff"}
                filter="url(#bar-glow)"
              />
            );
          })()}
      </svg>
    </div>
  );
}

// ─── Animated, Clickable Network SVG ─────────────────────────────────────────
function NetworkVisualization({ inputPixels, hidden1, hidden2, probabilities, prediction }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [signalPhase, setSignalPhase] = useState(0);
  const animRef = useRef(null);
  const startRef = useRef(null);

  const W = 1000;
  const H = 340;
  const TOP = 24;
  const BOT = H - 24;
  const COL = { input: 55, h1: 310, h2: 600, output: 870 };

  useEffect(() => {
    const animate = (ts) => {
      if (!startRef.current) startRef.current = ts;
      setSignalPhase(((ts - startRef.current) / 2200) % 1);
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  const INPUT_COUNT = 32;
  const H1_COUNT = 28;
  const H2_COUNT = 18;

  const inputNodes = useMemo(() => {
    const step = Math.floor(784 / INPUT_COUNT);
    return Array.from({ length: INPUT_COUNT }, (_, i) => ({
      value: inputPixels[i * step] ?? 0,
      y: TOP + (i / (INPUT_COUNT - 1)) * (BOT - TOP),
      idx: i * step,
    }));
  }, [inputPixels]);

  const h1Nodes = useMemo(() => {
    const step = Math.floor(128 / H1_COUNT);
    return Array.from({ length: H1_COUNT }, (_, i) => ({
      value: hidden1[i * step] ?? 0,
      y: TOP + (i / (H1_COUNT - 1)) * (BOT - TOP),
      idx: i * step,
    }));
  }, [hidden1]);

  const h2Nodes = useMemo(() => {
    const step = Math.floor(64 / H2_COUNT);
    return Array.from({ length: H2_COUNT }, (_, i) => ({
      value: hidden2[i * step] ?? 0,
      y: TOP + (i / (H2_COUNT - 1)) * (BOT - TOP),
      idx: i * step,
    }));
  }, [hidden2]);

  const outputNodes = useMemo(
    () =>
      Array.from({ length: 10 }, (_, i) => ({
        value: (probabilities[i] ?? 0) / 100,
        y: TOP + (i / 9) * (BOT - TOP),
        digit: i,
      })),
    [probabilities]
  );

  const connections = useMemo(() => {
    const lines = [];
    inputNodes.forEach((a, ai) => {
      h1Nodes.forEach((b, bi) => {
        if ((ai * 7 + bi * 3) % 11 === 0)
          lines.push({ x1: COL.input, y1: a.y, x2: COL.h1, y2: b.y, strength: b.value, layer: 0 });
      });
    });
    h1Nodes.forEach((a, ai) => {
      h2Nodes.forEach((b, bi) => {
        if ((ai * 5 + bi * 4) % 9 === 0)
          lines.push({ x1: COL.h1, y1: a.y, x2: COL.h2, y2: b.y, strength: b.value, layer: 1 });
      });
    });
    h2Nodes.forEach((a) => {
      outputNodes.forEach((b) => {
        lines.push({ x1: COL.h2, y1: a.y, x2: COL.output, y2: b.y, strength: b.value, layer: 2, digit: b.digit });
      });
    });
    return lines;
  }, [inputNodes, h1Nodes, h2Nodes, outputNodes]);

  const highlightSet = useMemo(() => {
    if (!selectedNode) return new Set();
    const set = new Set();
    const THRESH = 14;
    connections.forEach((c, i) => {
      const { layer } = selectedNode;
      if (layer === "input" && c.layer === 0 && Math.abs(c.y1 - selectedNode.y) < THRESH) set.add(i);
      if (layer === "h1" && c.layer === 0 && Math.abs(c.y2 - selectedNode.y) < THRESH) set.add(i);
      if (layer === "h1" && c.layer === 1 && Math.abs(c.y1 - selectedNode.y) < THRESH) set.add(i);
      if (layer === "h2" && c.layer === 1 && Math.abs(c.y2 - selectedNode.y) < THRESH) set.add(i);
      if (layer === "h2" && c.layer === 2 && Math.abs(c.y1 - selectedNode.y) < THRESH) set.add(i);
      if (layer === "output" && c.layer === 2 && Math.abs(c.y2 - selectedNode.y) < THRESH) set.add(i);
    });
    return set;
  }, [selectedNode, connections]);

  const handleNodeClick = useCallback((layer, node, cx) => {
    setSelectedNode((prev) =>
      prev?.layer === layer && prev?.y === node.y
        ? null
        : {
            layer,
            y: node.y,
            x: cx,
            value: node.value,
            label: node.digit !== undefined ? `Output digit ${node.digit}` : `${layer} · node ${node.idx}`,
          }
    );
  }, []);

  const signalDots = useMemo(() => {
    return connections
      .filter((c) => c.strength > 0.2)
      .map((c, i) => {
        const offset = (i * 0.13 + c.layer * 0.33) % 1;
        const t = (signalPhase + offset) % 1;
        const isOutputActive = c.digit === prediction && c.layer === 2;
        return {
          cx: c.x1 + (c.x2 - c.x1) * t,
          cy: c.y1 + (c.y2 - c.y1) * t,
          opacity: Math.sin(t * Math.PI) * (isOutputActive ? c.strength * 1.1 : c.strength * 0.75),
          isActive: isOutputActive,
          key: `sig-${i}`,
        };
      });
  }, [connections, signalPhase, prediction]);

  const NR = { input: 4, h1: 6, h2: 7, output: 9 };

  const renderNode = (layer, node, cx, r, baseColor, activeColor) => {
    const isSel = selectedNode?.layer === layer && selectedNode?.y === node.y;
    const isOutputActive = layer === "output" && node.digit === prediction;
    const color = isOutputActive ? activeColor : baseColor;
    const op = Math.max(0.12, node.value);
    const finalR = r + (isOutputActive ? 3 : 0) + (isSel ? 2 : 0);

    return (
      <g
        key={`${layer}-${node.idx ?? node.digit}`}
        style={{ cursor: "pointer" }}
        onClick={(e) => {
          e.stopPropagation();
          handleNodeClick(layer, node, cx);
        }}
      >
        {(isSel || isOutputActive) && (
          <circle
            cx={cx}
            cy={node.y}
            r={finalR + 2}
            fill="none"
            stroke={isOutputActive ? "rgba(0,255,140,0.5)" : "rgba(72,163,255,0.4)"}
            strokeWidth="1.5"
            className="pulse-ring"
          />
        )}
        <circle
          cx={cx}
          cy={node.y}
          r={finalR}
          fill={`rgba(${color},${op})`}
          stroke={isSel || isOutputActive ? `rgba(${color},0.85)` : "none"}
          strokeWidth="1.5"
          filter={isOutputActive ? "url(#glow-strong)" : node.value > 0.4 ? "url(#glow-blue)" : undefined}
        />
        {layer === "output" && (
          <text
            x={cx + 18}
            y={node.y + 4}
            fontSize={isOutputActive ? "13" : "11"}
            fontWeight={isOutputActive ? "700" : "400"}
            fill={isOutputActive ? "#00ff8c" : "rgba(255,255,255,0.48)"}
            fontFamily="'Courier New', monospace"
          >
            {node.digit}
            {node.value > 0.005 && (
              <tspan fontSize="10" fill={isOutputActive ? "rgba(0,255,140,0.8)" : "rgba(255,255,255,0.28)"}>
                {" "}
                {(node.value * 100).toFixed(1)}%
              </tspan>
            )}
          </text>
        )}
      </g>
    );
  };

  return (
    <div className="net-wrap" onClick={() => setSelectedNode(null)}>
      <div className="net-labels">
        {[
          { label: "Input", sub: "784 px", x: COL.input },
          { label: "Hidden 1", sub: "128 neurons", x: COL.h1 },
          { label: "Hidden 2", sub: "64 neurons", x: COL.h2 },
          { label: "Output", sub: "10 digits", x: COL.output },
        ].map(({ label, sub, x }) => (
          <span key={label} style={{ left: `${(x / W) * 100}%` }}>
            {label}
            <small>{sub}</small>
          </span>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        className="net-svg"
        onClick={(e) => e.stopPropagation()}
      >
        <defs>
          <filter id="glow-blue" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-strong" x="-150%" y="-150%" width="400%" height="400%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="bg-grad" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="#111824" />
            <stop offset="100%" stopColor="#080b0f" />
          </radialGradient>
          <style>{`
            @keyframes dash-flow { to { stroke-dashoffset: -20; } }
            @keyframes pulse-ring {
              0%   { r: 8px;  opacity: 0.8; }
              100% { r: 26px; opacity: 0; }
            }
            .connection-anim {
              stroke-dasharray: 5 5;
              animation: dash-flow 0.45s linear infinite;
            }
            .pulse-ring {
              animation: pulse-ring 1.4s ease-out infinite;
              transform-box: fill-box;
              transform-origin: center;
            }
          `}</style>
        </defs>

        <rect width={W} height={H} fill="url(#bg-grad)" rx="10" />

        {[COL.h1 - 122, COL.h2 - 122, COL.output - 122].map((x, i) => (
          <line
            key={i}
            x1={x}
            y1={TOP - 4}
            x2={x}
            y2={BOT + 4}
            stroke="rgba(255,255,255,0.035)"
            strokeWidth="1"
            strokeDasharray="2 8"
          />
        ))}

        {connections.map((c, i) => {
          const isHl = highlightSet.has(i);
          const isOutputActive = c.digit === prediction && c.layer === 2;
          const color = isOutputActive ? "0,255,140" : "72,163,255";
          const baseOp = selectedNode ? (isHl ? 0 : 0.012) : 0.025 + c.strength * 0.15;

          return (
            <g key={i}>
              <line x1={c.x1} y1={c.y1} x2={c.x2} y2={c.y2} stroke={`rgba(${color},${baseOp})`} strokeWidth="1" />
              {isHl && (
                <line
                  x1={c.x1}
                  y1={c.y1}
                  x2={c.x2}
                  y2={c.y2}
                  stroke={`rgba(${color},${0.18 + c.strength * 0.55})`}
                  strokeWidth={isOutputActive ? 2 : 1.5}
                  className="connection-anim"
                />
              )}
            </g>
          );
        })}

        {signalDots.map(
          (dot) =>
            dot.opacity > 0.04 && (
              <circle
                key={dot.key}
                cx={dot.cx}
                cy={dot.cy}
                r={dot.isActive ? 3 : 2.2}
                fill={`rgba(${dot.isActive ? "0,255,140" : "110,190,255"},${Math.min(dot.opacity, 0.95)})`}
                filter="url(#glow-blue)"
              />
            )
        )}

        {inputNodes.map((n) => renderNode("input", n, COL.input, NR.input, "72,163,255", "0,255,140"))}
        {h1Nodes.map((n) => renderNode("h1", n, COL.h1, NR.h1, "72,163,255", "0,255,140"))}
        {h2Nodes.map((n) => renderNode("h2", n, COL.h2, NR.h2, "100,185,255", "0,255,140"))}
        {outputNodes.map((n) => renderNode("output", n, COL.output, NR.output, "72,163,255", "0,255,140"))}

        {selectedNode &&
          (() => {
            const pct = (selectedNode.x / W) * 100;
            const flip = pct > 72;
            const tx = flip ? selectedNode.x - 145 : selectedNode.x + 18;
            const ty = Math.max(12, Math.min(H - 52, selectedNode.y - 22));
            return (
              <g>
                <rect
                  x={tx - 2}
                  y={ty - 2}
                  width={138}
                  height={46}
                  rx="7"
                  fill="rgba(8,14,22,0.93)"
                  stroke="rgba(72,163,255,0.35)"
                  strokeWidth="1"
                />
                <text
                  x={tx + 8}
                  y={ty + 13}
                  fontSize="10"
                  fill="rgba(255,255,255,0.45)"
                  fontFamily="'Courier New', monospace"
                >
                  {selectedNode.label}
                </text>
                <text
                  x={tx + 8}
                  y={ty + 30}
                  fontSize="13"
                  fontWeight="700"
                  fill="#48a3ff"
                  fontFamily="'Courier New', monospace"
                >
                  {(selectedNode.value * 100).toFixed(2)}% activation
                </text>
              </g>
            );
          })()}
      </svg>

      <div className="net-legend">
        Click any node to trace its connections · Moving dots = signal flow · Green = predicted digit
      </div>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────
function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(Array(10).fill(0));
  const [hidden1, setHidden1] = useState(Array(128).fill(0));
  const [hidden2, setHidden2] = useState(Array(64).fill(0));
  const [inputPixels, setInputPixels] = useState(Array(784).fill(0));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 30;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const getPointerPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches?.[0]?.clientX;
    const clientY = e.clientY ?? e.touches?.[0]?.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const { x, y } = getPointerPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const { x, y } = getPointerPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => setIsDrawing(false);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setProbabilities(Array(10).fill(0));
    setHidden1(Array(128).fill(0));
    setHidden2(Array(64).fill(0));
    setInputPixels(Array(784).fill(0));
    setError("");
  };

  const predictDigit = async () => {
    if (loading) return;

    const canvas = canvasRef.current;
    setLoading(true);
    setError("");

    try {
      const response = await axios.post(
        "https://digit-recognizer-api-916896635867.us-central1.run.app/predict",
        { image: canvas.toDataURL("image/png") },
        { timeout: 20000 }
      );

      setPrediction(response.data.prediction);
      setProbabilities(
        Array.isArray(response.data.probabilities) ? response.data.probabilities : Array(10).fill(0)
      );
      setHidden1(Array.isArray(response.data.hidden1) ? response.data.hidden1 : Array(128).fill(0));
      setHidden2(Array.isArray(response.data.hidden2) ? response.data.hidden2 : Array(64).fill(0));
      setInputPixels(Array.isArray(response.data.input_pixels) ? response.data.input_pixels : Array(784).fill(0));

      if (response.data?.error) {
        setError(response.data.error);
      }
    } catch (err) {
      console.error("Prediction error:", err);
      setError("Prediction failed or timed out. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Handwritten Digit Recognizer</h1>

      <div className="top-row">
        <div className="canvas-panel">
          <canvas
            ref={canvasRef}
            width={560}
            height={560}
            className="drawing-canvas large"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />

          <div className="buttons">
            <button onClick={predictDigit} disabled={loading}>
              {loading ? "Predicting..." : "Predict"}
            </button>
            <button onClick={clearCanvas} disabled={loading}>
              Clear
            </button>
          </div>

          {error && <p className="error-message">{error}</p>}

          {prediction !== null && (
            <h2>
              Guess: <span>{prediction}</span>
            </h2>
          )}
        </div>

        <div className="chart-card tall">
          <div className="chart-header">
            <span>Model confidence by digit</span>
            {prediction !== null && <span className="chart-prediction">Top guess: {prediction}</span>}
          </div>
          <div className="chart-area">
            <ConfidenceChart probabilities={probabilities} prediction={prediction} />
          </div>
        </div>
      </div>

      <div className="network-section">
        <h3>Neural Network Activity</h3>
        <NetworkVisualization
          inputPixels={inputPixels}
          hidden1={hidden1}
          hidden2={hidden2}
          probabilities={probabilities}
          prediction={prediction}
        />
      </div>
    </div>
  );
}

export default App;