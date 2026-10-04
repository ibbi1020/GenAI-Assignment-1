import { useEffect, useRef, useState } from "react";
import {
  CORRUPTIONS,
  SEVERITIES,
  generateSketch,
  loadHealth,
  loadSamples,
  restoreHard,
  restoreSoft,
  restoreUniversal,
} from "./api";
import { Task1Results, Task2Results, Task3Results } from "./Results";

const WORKSPACES = [
  {
    id: "universal",
    index: "01",
    name: "Universal Restoration",
    kicker: "One autoencoder, every kind of damage",
  },
  {
    id: "hard",
    index: "02",
    name: "Hard-Routed Restoration",
    kicker: "A classifier picks one specialist",
  },
  {
    id: "soft",
    index: "03",
    name: "Soft Mixture-of-Experts Restoration",
    kicker: "Four weights, one reconstruction",
  },
  {
    id: "sketch",
    index: "04",
    name: "Face-to-Sketch Generator",
    kicker: "Style 1, Style 2, or Style 3",
  },
];

export default function App() {
  const [workspace, setWorkspace] = useState(WORKSPACES[0]);
  const [health, setHealth] = useState(null);
  const [view, setView] = useState("results");
  const hasResults = workspace.id === "universal" || workspace.id === "hard" || workspace.id === "soft";

  useEffect(() => {
    loadHealth().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  return (
    <div className="relative mx-auto grid min-h-screen max-w-[1440px] grid-cols-1 lg:grid-cols-[280px_1fr]">
      <aside className="border-b border-white/10 px-6 py-8 lg:sticky lg:top-0 lg:h-screen lg:border-r lg:border-b-0">
        <p className="text-[11px] tracking-[0.28em] text-brass uppercase">Assignment 1</p>
        <h1 className="font-display mt-3 text-4xl leading-none text-paper">Four Plates</h1>
        <p className="mt-3 max-w-56 text-sm leading-relaxed text-mist">
          One darkroom. Four workspaces. Every result is 128 pixels square.
        </p>
        <nav className="mt-10 flex flex-col gap-1">
          {WORKSPACES.map((item, order) => {
            const active = item.id === workspace.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setWorkspace(item)}
                className={`rise flex items-baseline gap-3 rounded-sm px-2 py-3 text-left transition ${
                  active ? "bg-paper text-ink" : "text-paper/80 hover:bg-white/5"
                }`}
                style={{ animationDelay: `${order * 70}ms` }}
              >
                <span className="font-display text-sm text-brass">{item.index}</span>
                <span className="text-sm leading-snug">{item.name}</span>
              </button>
            );
          })}
        </nav>
        <p className="mt-10 text-xs tracking-wide text-mist">
          Backend {health?.status === "ok" ? "ready" : health ? "offline" : "checking"}
          {health?.models ? ` · ${Object.values(health.models).filter(Boolean).length} weights loaded` : ""}
        </p>
      </aside>
      <main className="px-6 py-8 lg:px-12 lg:py-10">
        <p className="text-[11px] tracking-[0.28em] text-brass uppercase">{workspace.index}</p>
        <h2 className="font-display mt-2 max-w-3xl text-5xl leading-[0.95] text-paper sm:text-6xl">
          {workspace.name}
        </h2>
        <p className="mt-4 text-base text-mist">{workspace.kicker}</p>
        {hasResults && (
          <div className="mt-8 flex gap-2" role="tablist">
            {[
              ["demo", "Try it"],
              ["results", "Results"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={view === id}
                onClick={() => setView(id)}
                className={`rounded-sm px-4 py-2 text-sm ${view === id ? "bg-paper text-ink" : "border border-white/15"}`}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        <div key={`${workspace.id}-${view}`} className="rise mt-10">
          {workspace.id === "universal" && view === "results" && <Task1Results />}
          {workspace.id === "hard" && view === "results" && <Task2Results />}
          {workspace.id === "soft" && view === "results" && <Task3Results />}
          {workspace.id === "universal" && view === "demo" && <RestoreBench run={restoreUniversal} showCorruption />}
          {workspace.id === "hard" && view === "demo" && <RestoreBench run={restoreHard} showCorruption showHard />}
          {workspace.id === "soft" && view === "demo" && <RestoreBench run={restoreSoft} showCorruption showSoft />}
          {workspace.id === "sketch" && <SketchBench />}
        </div>
      </main>
    </div>
  );
}

function RestoreBench({ run, showCorruption, showHard, showSoft }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [samples, setSamples] = useState([]);
  const [sampleId, setSampleId] = useState("");
  const [source, setSource] = useState("upload");
  const [corruption, setCorruption] = useState("salt_and_pepper");
  const [severity, setSeverity] = useState("medium");
  const [applyDamage, setApplyDamage] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    loadSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  function chooseFile(next) {
    setFile(next);
    setSampleId("");
    setSource("upload");
    setPreview(next ? URL.createObjectURL(next) : null);
    setResult(null);
  }

  function chooseSample(id) {
    setSampleId(id);
    setFile(null);
    setSource("sample");
    setApplyDamage(true);
    setPreview(`/samples/${id}`);
    setResult(null);
  }

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const payload = await run({
        file: source === "upload" ? file : null,
        sampleId: source === "sample" ? sampleId : "",
        corruption: applyDamage ? corruption : "as_uploaded",
        severity: applyDamage ? severity : "",
      });
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const ready = source === "upload" ? Boolean(file) : Boolean(sampleId);

  return (
    <div className="grid gap-8 xl:grid-cols-[320px_1fr]">
      <section className="space-y-6">
        <Dropzone onFile={chooseFile} preview={source === "upload" ? preview : null} />
        {samples.length > 0 && (
          <div>
            <p className="text-[11px] tracking-[0.22em] text-mist uppercase">Clean samples</p>
            <div className="mt-3 grid grid-cols-4 gap-2">
              {samples.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => chooseSample(sample.id)}
                  className={`overflow-hidden rounded-sm border ${
                    sampleId === sample.id ? "border-brass" : "border-white/10"
                  }`}
                >
                  <img src={sample.url} alt={sample.id} className="aspect-square w-full object-cover" />
                </button>
              ))}
            </div>
          </div>
        )}
        {showCorruption && (
          <fieldset className="space-y-3">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={applyDamage}
                onChange={(event) => setApplyDamage(event.target.checked)}
              />
              Apply a corruption before restoring
            </label>
            {applyDamage && (
              <div className="grid grid-cols-1 gap-3">
                <Select
                  label="Corruption"
                  value={corruption}
                  onChange={setCorruption}
                  options={CORRUPTIONS.map((item) => ({ value: item.id, label: item.label }))}
                />
                <Select
                  label="Severity"
                  value={severity}
                  onChange={setSeverity}
                  options={SEVERITIES.map((item) => ({ value: item, label: item }))}
                />
              </div>
            )}
          </fieldset>
        )}
        <button
          type="button"
          disabled={!ready || busy}
          onClick={submit}
          className="bg-safelight w-full rounded-sm px-4 py-3 text-sm font-medium text-paper disabled:opacity-40"
        >
          {busy ? "Running…" : "Run this plate"}
        </button>
        {error && <p className="text-sm text-safelight">{error}</p>}
      </section>
      <ResultStage result={result} showHard={showHard} showSoft={showSoft} />
    </div>
  );
}

function SketchBench() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [style, setStyle] = useState("Style 1");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (cameraOn && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [cameraOn]);

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    streamRef.current = stream;
    setCameraOn(true);
    setFile(null);
    setPreview(null);
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  function capture() {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      const shot = new File([blob], "webcam.png", { type: "image/png" });
      setFile(shot);
      setPreview(URL.createObjectURL(shot));
      stopCamera();
    }, "image/png");
  }

  async function submit() {
    setBusy(true);
    setError("");
    try {
      setResult(await generateSketch({ file, style }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-8 xl:grid-cols-[320px_1fr]">
      <section className="space-y-6">
        <Dropzone
          onFile={(next) => {
            setFile(next);
            setPreview(URL.createObjectURL(next));
            stopCamera();
          }}
          preview={preview}
        />
        <div className="space-y-3">
          {!cameraOn ? (
            <button type="button" onClick={startCamera} className="w-full rounded-sm border border-white/15 px-4 py-3 text-sm">
              Open webcam
            </button>
          ) : (
            <>
              <video ref={videoRef} autoPlay playsInline className="aspect-square w-full rounded-sm bg-black object-cover" />
              <button type="button" onClick={capture} className="w-full rounded-sm border border-brass px-4 py-3 text-sm">
                Capture frame
              </button>
            </>
          )}
        </div>
        <div>
          <p className="text-[11px] tracking-[0.22em] text-mist uppercase">Style</p>
          <div className="mt-3 grid grid-cols-3 gap-2">
            {["Style 1", "Style 2", "Style 3"].map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => setStyle(name)}
                className={`rounded-sm px-2 py-3 text-sm ${
                  style === name ? "bg-paper text-ink" : "border border-white/15"
                }`}
              >
                {name}
              </button>
            ))}
          </div>
        </div>
        <button
          type="button"
          disabled={!file || busy}
          onClick={submit}
          className="bg-safelight w-full rounded-sm px-4 py-3 text-sm font-medium disabled:opacity-40"
        >
          {busy ? "Drawing…" : "Generate sketch"}
        </button>
        {error && <p className="text-sm text-safelight">{error}</p>}
      </section>
      <div className="grid gap-4 md:grid-cols-2">
        <Plate title="Photograph" src={result?.input_png || preview} />
        <Plate title="Sketch" src={result?.output_png} downloadName={`${style.replace(" ", "-").toLowerCase()}-sketch.png`} />
        <Meta result={result} extra={result ? `Style ${result.style}` : style} />
      </div>
    </div>
  );
}

function ResultStage({ result, showHard, showSoft }) {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Plate title="Input" src={result?.input_png} />
        <Plate title="Restored" src={result?.output_png} downloadName="restored.png" />
      </div>
      <Meta result={result} />
      {showHard && result?.probabilities && <WeightList title="Classifier probabilities" values={result.probabilities} highlight={result.predicted_class} />}
      {showHard && result && (
        <p className="text-sm text-paper/80">
          Predicted class: <span className="text-paper">{result.predicted_class || "—"}</span>
          {" · "}
          Chosen expert: <span className="text-paper">{result.chosen_expert || "—"}</span>
        </p>
      )}
      {showSoft && result?.weights && (
        <>
          <WeightList title="Expert weights" values={result.weights} highlight={result.dominant_experts} />
          <p className="text-sm text-mist">
            Weights sum to{" "}
            <span className="text-paper tabular-nums">
              {Object.values(result.weights)
                .reduce((sum, value) => sum + Number(value), 0)
                .toFixed(2)}
            </span>
            . Highlighted branches contributed at least a quarter of the mix.
          </p>
        </>
      )}
      {showSoft && result && !result.weights && (
        <p className="text-sm text-mist">Expert weights show up here once the mixture weights are loaded.</p>
      )}
    </div>
  );
}

function WeightList({ title, values, highlight }) {
  const entries = Object.entries(values);
  const highlights = Array.isArray(highlight) ? highlight : [highlight];
  return (
    <div>
      <p className="text-[11px] tracking-[0.22em] text-mist uppercase">{title}</p>
      <ul className="mt-3 space-y-2">
        {entries.map(([name, value]) => (
          <li key={name} className="grid grid-cols-[140px_1fr_48px] items-center gap-3 text-sm">
            <span className={highlights.includes(name) ? "text-paper" : "text-mist"}>{labelFor(name)}</span>
            <span className="h-1.5 overflow-hidden rounded-full bg-white/10">
              <span
                className={`block h-full ${highlights.includes(name) ? "bg-safelight" : "bg-brass"}`}
                style={{ width: `${Math.max(0, Math.min(1, Number(value))) * 100}%` }}
              />
            </span>
            <span className="text-right tabular-nums">{Number(value).toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Meta({ result, extra }) {
  const settings = result?.corruption_settings;
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-mist">
      <span>Inference {result ? `${result.inference_ms} ms` : "—"}</span>
      <span>{result?.model_loaded ? "Weights loaded" : result ? "Weights not loaded yet" : "Waiting for a run"}</span>
      {extra && <span>{extra}</span>}
      {settings && settings.corruption !== "as_uploaded" && (
        <span>
          {labelFor(settings.corruption)} · {settings.severity}
          {settings.salt_probability != null ? ` · p=${settings.salt_probability}` : ""}
          {settings.blur_kernel != null ? ` · kernel ${settings.blur_kernel}, σ ${settings.blur_sigma}` : ""}
          {settings.coverage != null ? ` · coverage ${Math.round(settings.coverage * 100)}%` : ""}
        </span>
      )}
    </div>
  );
}

function Plate({ title, src, downloadName }) {
  return (
    <figure className="bg-rebate rounded-sm p-3">
      <figcaption className="mb-2 flex items-center justify-between text-[11px] tracking-[0.22em] text-brass uppercase">
        <span>{title}</span>
        {src && downloadName && (
          <a href={src} download={downloadName} className="tracking-normal text-paper/80 normal-case">
            Download
          </a>
        )}
      </figcaption>
      <div className="bg-paper flex aspect-square items-center justify-center overflow-hidden">
        {src ? (
          <img src={src} alt={title} className="h-full w-full object-contain" />
        ) : (
          <span className="font-display text-2xl text-ink/30">Empty plate</span>
        )}
      </div>
    </figure>
  );
}

function Dropzone({ onFile, preview }) {
  return (
    <label className="border-paper/20 hover:border-brass block cursor-pointer rounded-sm border border-dashed p-4">
      <input
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(event) => {
          const next = event.target.files?.[0];
          if (next) onFile(next);
        }}
      />
      {preview ? (
        <img src={preview} alt="Selected upload" className="mx-auto aspect-square max-h-72 w-full rounded-sm object-cover" />
      ) : (
        <span className="flex h-40 items-center justify-center text-center text-sm text-mist">
          Drop a photo here, or click to upload
        </span>
      )}
    </label>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="block text-sm">
      <span className="text-[11px] tracking-[0.22em] text-mist uppercase">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="border-white/15 mt-1 w-full rounded-sm border bg-transparent px-3 py-2"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value} className="text-ink">
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function labelFor(name) {
  return (
    {
      clean: "Clean",
      salt_and_pepper: "Salt",
      gaussian_blur: "Blur",
      rectangular_occlusion: "Occlusion",
      as_uploaded: "As uploaded",
    }[name] || name
  );
}
