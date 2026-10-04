const CORRUPTIONS = [
  { id: "salt_and_pepper", label: "Salt and pepper" },
  { id: "gaussian_blur", label: "Gaussian blur" },
  { id: "rectangular_occlusion", label: "Rectangular occlusion" },
];

const SEVERITIES = ["low", "medium", "high"];

async function postImage(path, { file, sampleId, corruption, severity, style }) {
  const body = new FormData();
  if (file) body.append("file", file);
  if (sampleId) body.append("sample_id", sampleId);
  if (corruption) body.append("corruption", corruption);
  if (severity) body.append("severity", severity);
  if (style) body.append("style", style);
  const response = await fetch(path, { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "The backend rejected this image.");
  }
  return payload;
}

export function restoreUniversal(options) {
  return postImage("/universal-restoration", options);
}

export function restoreHard(options) {
  return postImage("/hard-routing", options);
}

export function restoreSoft(options) {
  return postImage("/soft-mixture", options);
}

export function generateSketch(options) {
  return postImage("/face-to-sketch", options);
}

export async function loadHealth() {
  const response = await fetch("/health");
  if (!response.ok) throw new Error("Backend is not answering.");
  return response.json();
}

export async function loadSamples() {
  const response = await fetch("/samples");
  if (!response.ok) return [];
  const payload = await response.json();
  return payload.samples || [];
}

export async function loadResults(task) {
  const response = await fetch(`/results/${task}`);
  if (!response.ok) throw new Error("Could not load results from the backend.");
  return response.json();
}

export { CORRUPTIONS, SEVERITIES };
