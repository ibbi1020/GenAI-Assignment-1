import { useEffect, useState } from "react";
import { loadResults } from "./api";

const CORRUPTION_ORDER = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"];
const NAMES = {
  clean: "Clean",
  salt_and_pepper: "Salt and pepper",
  gaussian_blur: "Gaussian blur",
  rectangular_occlusion: "Rectangular occlusion",
  all: "All",
};

const SHORT = { clean: "Clean", salt_and_pepper: "Salt", gaussian_blur: "Blur", rectangular_occlusion: "Occl." };

function useResults(task) {
  const [state, setState] = useState({ data: null, error: "" });
  useEffect(() => {
    let alive = true;
    loadResults(task)
      .then((data) => alive && setState({ data, error: "" }))
      .catch((error) => alive && setState({ data: null, error: error.message }));
    return () => {
      alive = false;
    };
  }, [task]);
  return state;
}

export function Task1Results() {
  const { data, error } = useResults("task1");
  if (error) return <p className="text-sm text-safelight">{error}</p>;
  if (!data) return <p className="text-sm text-mist">Loading…</p>;
  if (!data.validation_vs_input) {
    return <p className="text-sm text-mist">No Task 1 evaluation file found yet.</p>;
  }
  const rows = CORRUPTION_ORDER.map((name) => ({ name, ...data.validation_vs_input[name] }));
  const params = data.params;
  const weak = rows.filter((row) => row.name !== "clean" && !Object.values(row.beats_input).every(Boolean));
  return (
    <div className="space-y-10">
      <Note>
        &ldquo;Input&rdquo; is the corrupted image with no restoration, so a restored score is only good if it beats
        that line. The test table is the final result. The validation table was used while training.
      </Note>

      {data.test_vs_input && <TestTable rows={data.test_vs_input} />}

      <section>
        <Heading>Validation: restored vs. corrupted input</Heading>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
              <tr>
                <th className="py-2 pr-4 font-normal">Corruption</th>
                <th className="py-2 pr-4 font-normal">Images</th>
                <MetricHead label="L1 ↓" />
                <MetricHead label="SSIM ↑" />
                <MetricHead label="PSNR ↑" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name} className="border-t border-white/10 align-top">
                  <td className="py-3 pr-4 text-paper">{NAMES[row.name]}</td>
                  <td className="py-3 pr-4 text-mist tabular-nums">{row.restored.count}</td>
                  <MetricCell row={row} metric="l1" digits={4} />
                  <MetricCell row={row} metric="ssim" digits={3} />
                  <MetricCell row={row} metric="psnr" digits={1} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-mist">
          Gold cells with ✓ beat the corrupted input; ✗ means they do not. A clean image is its own perfect copy, so it
          cannot be beaten.
          {weak.length > 0 && ` Not yet better than the input on every metric: ${weak.map((row) => NAMES[row.name]).join(", ")}.`}
        </p>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <ImagePlate title="Validation examples" src={data.images.val_preview} />
        <ImagePlate title="Training curves" src={data.images.loss_curves} />
      </div>

      {params && (
        <section>
          <Heading>Model and hyperparameters</Heading>
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {[
              ["Bottleneck map", `${params.bottleneck_dimension}×${params.bottleneck_dimension}`],
              ["First-layer channels", params.encoder_channels],
              ["Norm", params.norm],
              ["Loss mix α", params.alpha],
              ["Learning rate", params.lr],
              ["Batch size", params.batch_size],
            ].map(([label, value]) => (
              <div key={label}>
                <dt className="text-[11px] tracking-[0.18em] text-mist uppercase">{label}</dt>
                <dd className="text-paper tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
    </div>
  );
}

export function Task2Results() {
  const { data, error } = useResults("task2");
  if (error) return <p className="text-sm text-safelight">{error}</p>;
  if (!data) return <p className="text-sm text-mist">Loading…</p>;
  const classifier = data.classifier;
  return (
    <div className="space-y-10">
      {classifier ? <ClassifierSection classifier={classifier} classes={data.classes} /> : (
        <p className="text-sm text-mist">No classifier results found yet.</p>
      )}
      <SpecialistSection specialists={data.specialists} />
      <RoutingSection routing={data.routing} />
      {(data.images.predicted_examples || data.images.routing_failures) && (
        <div className="grid gap-4 md:grid-cols-2">
          <ImagePlate title="Predicted-routing examples" src={data.images.predicted_examples} />
          <ImagePlate title="Routing failures" src={data.images.routing_failures} />
        </div>
      )}
    </div>
  );
}

function ClassifierSection({ classifier, classes }) {
  const test = classifier.test_metrics;
  const params = classifier.selected_params;
  return (
    <>
      <Note>
        Classifier scores are on the <b>test</b> split (hyperparameters were chosen on validation only).
      </Note>
      <section>
        <Heading>Corruption classifier</Heading>
        <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Accuracy" value={test.accuracy} />
          <Stat label="Macro F1" value={test.macro_f1} />
          <Stat label="Macro precision" value={test.macro_precision} />
          <Stat label="Macro recall" value={test.macro_recall} />
        </div>
      </section>

      <section className="grid gap-8 xl:grid-cols-2">
        <div>
          <Heading>Per-class scores</Heading>
          <table className="mt-3 w-full text-sm">
            <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
              <tr>
                <th className="py-2 font-normal">Class</th>
                <th className="py-2 text-right font-normal">Precision</th>
                <th className="py-2 text-right font-normal">Recall</th>
                <th className="py-2 text-right font-normal">F1</th>
              </tr>
            </thead>
            <tbody>
              {test.per_class.map((row) => (
                <tr key={row.class} className="border-t border-white/10">
                  <td className="py-2 text-paper">{NAMES[row.class]}</td>
                  <td className="py-2 text-right tabular-nums">{row.precision.toFixed(3)}</td>
                  <td className="py-2 text-right tabular-nums">{row.recall.toFixed(3)}</td>
                  <td className="py-2 text-right tabular-nums">{row.f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <Heading>Confusion matrix (rows: true, columns: predicted)</Heading>
          <ConfusionGrid matrix={test.normalized_confusion} classes={classes} />
        </div>
      </section>

      <section>
        <Heading>Accuracy by severity</Heading>
        <SeverityTable rows={classifier.accuracy_by_severity} />
        <p className="mt-3 text-xs text-mist">
          Cells below 90% are highlighted. Low-severity damage is the hardest to spot because it looks closest to a
          clean image.
        </p>
      </section>

      <section>
        <Heading>Selected hyperparameters</Heading>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
          {Object.entries(params)
            .filter(([key]) => key !== "train_seed")
            .map(([key, value]) => (
              <div key={key}>
                <dt className="text-[11px] tracking-[0.18em] text-mist uppercase">{key.replaceAll("_", " ")}</dt>
                <dd className="text-paper tabular-nums">{value}</dd>
              </div>
            ))}
        </dl>
      </section>
    </>
  );
}

function SpecialistSection({ specialists }) {
  if (!specialists) {
    return (
      <section>
        <Heading>Specialist autoencoders</Heading>
        <p className="mt-3 text-sm text-mist">Specialists are still training. Results appear here when they finish.</p>
      </section>
    );
  }
  const entries = Object.entries(specialists.specialists || {});
  return (
    <section>
      <Heading>Specialist autoencoders</Heading>
      <table className="mt-3 w-full text-sm">
        <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
          <tr>
            <th className="py-2 font-normal">Specialist</th>
            <th className="py-2 text-right font-normal">Best val score ↓</th>
            <th className="py-2 text-right font-normal">ONNX diff</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, info]) => (
            <tr key={name} className="border-t border-white/10">
              <td className="py-2 text-paper">{NAMES[name] || name}</td>
              <td className="py-2 text-right tabular-nums">{Number(info.best_objective).toFixed(4)}</td>
              <td className="py-2 text-right tabular-nums">{Number(info.onnx_max_abs_diff).toExponential(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function RoutingSection({ routing }) {
  if (!routing) {
    return (
      <section>
        <Heading>Routing on the test split</Heading>
        <p className="mt-3 text-sm text-mist">
          Oracle vs. predicted routing is scored after the specialists finish.
        </p>
      </section>
    );
  }
  const systems = [
    ["identity", "Do nothing"],
    ["universal", "Task 1 autoencoder"],
    ["oracle", "Oracle routing"],
    ["predicted", "Predicted routing"],
  ].filter(([key]) => routing[key]);
  const pick = (key, corruption) => {
    const rows = routing[key].filter((row) => row.corruption === corruption);
    if (!rows.length) return null;
    const count = rows.reduce((sum, row) => sum + row.count, 0);
    const mean = (metric) => rows.reduce((sum, row) => sum + row[metric] * row.count, 0) / count;
    return { l1: mean("l1"), ssim: mean("ssim"), psnr: mean("psnr") };
  };
  return (
    <section>
      <Heading>Routing on the test split</Heading>
      <p className="mt-1 text-sm text-mist">
        Misrouted images: {(routing.misroute_rate * 100).toFixed(1)}%. Oracle uses the true label; predicted uses the
        classifier.
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
            <tr>
              <th className="py-2 font-normal">Corruption</th>
              {systems.map(([key, label]) => (
                <th key={key} className="py-2 text-right font-normal">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...CORRUPTION_ORDER, "all"].map((corruption) => (
              <tr key={corruption} className="border-t border-white/10">
                <td className="py-2 text-paper">{NAMES[corruption]}</td>
                {systems.map(([key]) => {
                  const cell = pick(key, corruption);
                  return (
                    <td key={key} className="py-2 text-right tabular-nums">
                      {cell ? `${cell.l1.toFixed(4)} · ${cell.ssim.toFixed(3)} · ${cell.psnr.toFixed(1)}` : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-mist">Each cell is L1 · SSIM · PSNR (lower L1 is better; higher SSIM and PSNR are better).</p>
    </section>
  );
}

export function Task3Results() {
  const { data, error } = useResults("task3");
  if (error) return <p className="text-sm text-safelight">{error}</p>;
  if (!data) return <p className="text-sm text-mist">Loading…</p>;
  const summary = data.summary;
  return (
    <div className="space-y-10">
      <Note>
        The gate starts as a copy of the Task 2 classifier. The three specialists start as the Task 2
        autoencoders. Clean is an identity branch that returns the input. Warm-up trains only the gate, then
        joint fine-tuning unfreezes the experts at a smaller learning rate.
        {summary
          ? " The weight table is the test split. The saved weights are the epoch with the best validation score."
          : " Training has not been run, so the tables below are empty."}
      </Note>

      <section>
        <Heading>Average gate weights</Heading>
        {summary?.gate_weights ? (
          <GateWeightTable rows={summary.gate_weights} />
        ) : (
          <p className="mt-3 text-sm text-mist">
            Mean weight for each branch, split by true corruption and test severity, lands here.
          </p>
        )}
      </section>

      <section>
        <Heading>Expert activity</Heading>
        {summary?.expert_activity ? (
          <ExpertActivity activity={summary.expert_activity} />
        ) : (
          <p className="mt-3 text-sm text-mist">
            Inactive experts, and experts that take another corruption&apos;s images, land here.
          </p>
        )}
      </section>

      <section>
        <Heading>Reconstruction</Heading>
        {summary?.reconstruction ? (
          <ReconstructionTable rows={summary.reconstruction} />
        ) : (
          <p className="mt-3 text-sm text-mist">L1, SSIM, and PSNR for the mixture land here.</p>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-3">
        <ImagePlate title="Gate-weight heatmap" src={data.images.weight_heatmap} />
        <ImagePlate title="One expert dominates" src={data.images.dominant_examples} />
        <ImagePlate title="Weight is shared" src={data.images.shared_examples} />
      </div>

      <section>
        <Heading>Selected hyperparameters</Heading>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
          {(summary?.selected_params
            ? Object.entries(summary.selected_params).filter(([key]) => key !== "train_seed")
            : [
                ["temperature", 5],
                ["batch size", 64],
                ["warmup lr", "1e-3"],
                ["finetune lr", "1e-4"],
                ["lambda l1", 0.8],
                ["lambda ssim", 0.1],
                ["lambda cls", 0.01],
                ["lambda balance", 0.01],
                ["warmup epochs", 20],
                ["epochs", 200],
                ["patience", 20],
              ]
          ).map(([key, value]) => (
            <div key={key}>
              <dt className="text-[11px] tracking-[0.18em] text-mist uppercase">{String(key).replaceAll("_", " ")}</dt>
              <dd className="text-paper tabular-nums">{formatParam(value)}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-xs text-mist">
          {summary?.selected_params
            ? `Warm-up was ${summary.selected_params.warmup_epochs} epochs at ${summary.selected_params.warmup_lr}. Fine-tuning ran ${summary.selected_params.epochs} epochs at ${summary.selected_params.finetune_lr}. Best validation score ${Number(summary.validation_objective).toFixed(4)} (lower is better).`
            : "Warm-up trains only the gate. Fine-tuning then trains the gate and the three specialists together."}
        </p>
      </section>
    </div>
  );
}

function GateWeightTable({ rows }) {
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full min-w-[640px] text-sm">
        <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
          <tr>
            <th className="py-2 pr-3 font-normal">Corruption</th>
            <th className="py-2 pr-3 font-normal">Severity</th>
            <th className="py-2 pr-3 text-right font-normal">Images</th>
            {CORRUPTION_ORDER.map((name) => (
              <th key={name} className="py-2 pr-3 text-right font-normal">
                {SHORT[name]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.corruption}-${row.severity ?? "none"}`} className="border-t border-white/10">
              <td className="py-2 pr-3 text-paper">{NAMES[row.corruption] || row.corruption}</td>
              <td className="py-2 pr-3 text-mist">{row.severity || "—"}</td>
              <td className="py-2 pr-3 text-right tabular-nums">{row.count}</td>
              {CORRUPTION_ORDER.map((name) => (
                <td key={name} className="py-2 pr-3 text-right tabular-nums">
                  {Number(row.weights[name]).toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExpertActivity({ activity }) {
  const inactive = activity.inactive || [];
  const foreign = activity.takes_foreign_inputs || [];
  return (
    <div className="mt-3 space-y-4">
      <p className="text-sm text-mist">
        Inactive (mean weight under 0.05):{" "}
        <span className="text-paper">{inactive.length ? inactive.map((name) => NAMES[name] || name).join(", ") : "none"}</span>
      </p>
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {CORRUPTION_ORDER.map((name) => (
          <div key={name}>
            <dt className="text-[11px] tracking-[0.18em] text-mist uppercase">{SHORT[name]} mean</dt>
            <dd className="text-paper tabular-nums">{Number(activity.mean_weights[name]).toFixed(3)}</dd>
          </div>
        ))}
      </dl>
      {foreign.length === 0 ? (
        <p className="text-sm text-mist">No expert&apos;s mean weight beats the matching branch on a true class.</p>
      ) : (
        <ul className="space-y-2 text-sm">
          {foreign.map((item) => (
            <li key={`${item.true_corruption}-${item.expert}`} className="text-paper/90">
              On {NAMES[item.true_corruption]}, {NAMES[item.expert]} averages{" "}
              {Number(item.mean_weight_on_this_expert).toFixed(2)} against the matching branch&apos;s{" "}
              {Number(item.mean_weight_on_true_expert).toFixed(2)}.
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ReconstructionTable({ rows }) {
  return (
    <table className="mt-3 w-full text-sm">
      <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
        <tr>
          <th className="py-2 font-normal">Corruption</th>
          <th className="py-2 font-normal">Severity</th>
          <th className="py-2 text-right font-normal">L1 ↓</th>
          <th className="py-2 text-right font-normal">SSIM ↑</th>
          <th className="py-2 text-right font-normal">PSNR ↑</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={`${row.corruption}-${row.severity ?? "all"}`} className="border-t border-white/10">
            <td className="py-2 text-paper">{NAMES[row.corruption] || row.corruption}</td>
            <td className="py-2 text-mist">{row.severity || "—"}</td>
            <td className="py-2 text-right tabular-nums">{Number(row.l1).toFixed(4)}</td>
            <td className="py-2 text-right tabular-nums">{Number(row.ssim).toFixed(3)}</td>
            <td className="py-2 text-right tabular-nums">{Number(row.psnr).toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatParam(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? value : value.toPrecision(3);
  }
  return value;
}

function SeverityTable({ rows }) {
  const byCorruption = {};
  for (const row of rows) {
    (byCorruption[row.corruption] ||= {})[row.severity ?? "none"] = row.accuracy;
  }
  const cell = (value) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`);
  return (
    <table className="mt-3 w-full max-w-xl text-sm">
      <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
        <tr>
          <th className="py-2 font-normal">Corruption</th>
          <th className="py-2 text-right font-normal">Low</th>
          <th className="py-2 text-right font-normal">Medium</th>
          <th className="py-2 text-right font-normal">High</th>
        </tr>
      </thead>
      <tbody>
        {CORRUPTION_ORDER.map((name) => {
          const entry = byCorruption[name] || {};
          return (
            <tr key={name} className="border-t border-white/10">
              <td className="py-2 text-paper">{NAMES[name]}</td>
              {name === "clean" ? (
                <td colSpan={3} className="py-2 text-right tabular-nums">
                  {cell(entry.none)}
                </td>
              ) : (
                ["low", "medium", "high"].map((level) => (
                  <td
                    key={level}
                    className={`py-2 text-right tabular-nums ${entry[level] < 0.9 ? "text-safelight" : ""}`}
                  >
                    {cell(entry[level])}
                  </td>
                ))
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ConfusionGrid({ matrix, classes }) {
  return (
    <div className="mt-3 grid grid-cols-[96px_repeat(4,1fr)] gap-1 text-xs">
      <span />
      {classes.map((name) => (
        <span key={name} className="truncate text-center text-mist">
          {SHORT[name]}
        </span>
      ))}
      {matrix.map((row, i) => (
        <ConfusionRow key={classes[i]} label={NAMES[classes[i]]} row={row} diagonal={i} />
      ))}
    </div>
  );
}

function ConfusionRow({ label, row, diagonal }) {
  return (
    <>
      <span className="self-center truncate text-mist">{label}</span>
      {row.map((value, j) => (
        <span
          key={j}
          className="rounded-sm py-3 text-center tabular-nums text-paper"
          style={{
            backgroundColor:
              j === diagonal
                ? `rgba(201, 162, 39, ${0.12 + value * 0.6})`
                : `rgba(214, 69, 44, ${Math.min(1, value * 4) * 0.7})`,
          }}
        >
          {(value * 100).toFixed(1)}%
        </span>
      ))}
    </>
  );
}

function TestTable({ rows }) {
  const beats = (row, metric) =>
    metric === "l1" ? row.restored.l1 < row.input.l1 : row.restored[metric] > row.input[metric];
  const digits = { l1: 4, ssim: 3, psnr: 1 };
  return (
    <section>
      <Heading>Test: restored vs. corrupted input</Heading>
      <p className="mt-1 text-sm text-mist">
        36,690 images (3,669 photos × 10 conditions). Each severity row is 3,669 images.
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="text-left text-[11px] tracking-[0.18em] text-mist uppercase">
            <tr>
              <th className="py-2 pr-4 font-normal">Corruption</th>
              <th className="py-2 pr-4 font-normal">Severity</th>
              <MetricHead label="L1 ↓" />
              <MetricHead label="SSIM ↑" />
              <MetricHead label="PSNR ↑" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isClean = row.corruption === "clean";
              const first = isClean || row.severity === "low";
              return (
                <tr
                  key={`${row.corruption}-${row.severity}`}
                  className={`align-top ${first ? "border-t border-white/10" : ""}`}
                >
                  <td className="py-2 pr-4 text-paper">{first ? NAMES[row.corruption] : ""}</td>
                  <td className="py-2 pr-4 text-mist">{isClean ? "—" : row.severity === "all" ? "mean" : row.severity}</td>
                  {["l1", "ssim", "psnr"].map((metric) => (
                    <td key={metric} className="py-2 pr-4 tabular-nums">
                      <span className={!isClean && beats(row, metric) ? "text-brass" : "text-paper"}>
                        {row.restored[metric].toFixed(digits[metric])}
                      </span>
                      <span className="ml-2 text-xs text-mist">in {row.input[metric].toFixed(digits[metric])}</span>
                      {!isClean && <span className="ml-2 text-xs">{beats(row, metric) ? "✓" : "✗"}</span>}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-mist">
        ✓ means the restored image beats the corrupted input. Low blur is the main loss: it is already close to the
        clean image.
      </p>
    </section>
  );
}

function MetricHead({ label }) {
  return <th className="py-2 pr-4 font-normal">{label}</th>;
}

function MetricCell({ row, metric, digits }) {
  const beats = row.beats_input?.[metric];
  const isClean = row.name === "clean";
  return (
    <td className="py-3 pr-4 tabular-nums">
      <span className={beats && !isClean ? "text-brass" : "text-paper"}>{row.restored[metric].toFixed(digits)}</span>
      <span className="ml-2 text-xs text-mist">in {row.input[metric].toFixed(digits)}</span>
      {!isClean && <span className="ml-2 text-xs">{beats ? "✓" : "✗"}</span>}
    </td>
  );
}

function Stat({ label, value }) {
  return (
    <div className="bg-rebate rounded-sm p-4">
      <p className="text-[11px] tracking-[0.18em] text-mist uppercase">{label}</p>
      <p className="font-display mt-1 text-3xl text-paper tabular-nums">{(value * 100).toFixed(1)}%</p>
    </div>
  );
}

function ImagePlate({ title, src }) {
  return (
    <figure className="bg-rebate rounded-sm p-3">
      <figcaption className="mb-2 text-[11px] tracking-[0.22em] text-brass uppercase">{title}</figcaption>
      {src ? (
        <img src={src} alt={title} className="w-full rounded-sm bg-paper" />
      ) : (
        <div className="text-ink/40 bg-paper flex aspect-video items-center justify-center text-sm">Not generated yet</div>
      )}
    </figure>
  );
}

function Heading({ children }) {
  return <h3 className="font-display text-2xl text-paper">{children}</h3>;
}

function Note({ children }) {
  return <p className="border-brass/60 border-l-2 pl-3 text-sm text-mist">{children}</p>;
}
