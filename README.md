# ECE 226 — Benchmarking Methodologies & Profiling Ecosystems

**Group 23** — Shahid Mulla, Cheng-Lun Tai, Subhon Ghosh  
**Course:** ECE 226, Winter 2026, UC San Diego  
**Instructor:** Prof. Farinaz Koushanfar

---

## What This Is

A benchmarking study of three LLM kernels (GEMM, Element-wise Add, Softmax) on GPU using the Roofline Model. We profile each kernel across problem sizes and precisions (FP32/FP16) to classify them as compute-bound or memory-bound and derive optimization strategies.

---

## Repo Structure

```
.
├── ECE226_roofline_unified.ipynb       # Main notebook — run this to reproduce all results
├── roofline_model.ipynb                # Roofline construction
├── kernel_mapping.ipynb                # Kernel OI & performance mapping
├── roofline_benchmarking.ipynb         # Kernel benchmarking & analysis
├── ECE_226_Final_Report.pdf            # Final report
├── group23_benchmarking_presentation.pptx
└── Results/                            # All output plots and CSVs
```

---

## How to Reproduce Results

1. Open `ECE226_roofline_unified.ipynb` in **Google Colab**
2. Set runtime to **GPU** (L4 recommended): `Runtime → Change runtime type → GPU`
3. `Runtime → Run all`

All plots and result CSVs will be saved automatically to the working directory.

---

## Results Folder

Contains all generated figures and data from the notebook run, including roofline plots, latency curves, kernel performance comparisons, Nsight Systems-equivalent timeline plots, and classification tables.
