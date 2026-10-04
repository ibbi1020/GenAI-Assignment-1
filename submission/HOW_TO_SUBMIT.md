# What to upload

Google Classroom asks for the PDF only. Upload:

`submission/report/main.pdf`

The brief does not ask for a special file name. Upload that PDF by itself. Do not upload this folder, the code, the datasets, or a video file. The GitHub link inside the PDF is https://github.com/ibbi1020/GenAI-Assignment-1.

Before you upload, put your name, course, and student ID on the first page. The GitHub URL is already in the PDF.

## Left out on purpose

- Task 4 (face-to-sketch) is not trained. The app still has that workspace. It will not draw a sketch.
- The YouTube demo is not recorded. The paper says so. Do not attach a video file.

## What the PDF already contains

Tasks 1–3: data, architectures, losses, training, Optuna where it was actually used, test tables, the Task 3 routing heatmap, and the AI-use appendix.

Figures live next to the source in `submission/report/figures/`. Rebuild with:

```bash
cd submission/report && tectonic main.tex
```
