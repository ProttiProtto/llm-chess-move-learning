# NVFP4 Serving Notes

## Measured Configuration

The v2 experiments used NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0), PyTorch 2.13.0+cu132, vLLM 0.27.0, and Transformers 5.14.1. This documents the tested configuration, not a universal list of supported GPUs or future dependency versions.

Cell 3 of the publication-quality and parallelism notebooks installs/verifies the serving stack and CUDA JIT components. Cell 4 runs the evaluation or benchmark; Cell 5 only displays saved results. Use a fresh runtime for serving rather than mixing it with the export environment.

## Export and Backend

The publication export is ModelOpt NVFP4 W4A4, not the optional weight-only NVFP4A16 route. Check the saved quantization metadata, including both weight and input-activation bit widths, excluded modules, scales, and export validation results.

The serving configuration requests `flashinfer_cutlass` for dense linear layers. The separately configured `flashinfer_b12x` MoE option does not establish that these models executed MoE kernels. Backend flags and quantization metadata are evidence of configuration; only profiling could establish detailed kernel utilization.

## Failure Guide

| Symptom | Check |
| --- | --- |
| CUDA compiler/header mismatch | Preserve the error and rerun the pinned serving setup in a fresh runtime; avoid mixing CUDA component versions. |
| Missing B12x kernel for dense linears | Keep dense backend FlashInfer CUTLASS, not the MoE backend. |
| Cannot find libcudart or libnvrtc | Inspect Cell 3's CUDA JIT/toolkit validation and library search paths. |
| Failure after model loading | Save worker/server logs, checkpoint manifest, package versions and GPU details before changing the stack. |
| Out of memory | Reduce context/concurrency and inspect KV reservation; changing settings invalidates a direct speed comparison. |
| Poor outputs despite successful execution | Compare the same checkpoint family against BF16 on the same quality contract. Successful kernel execution does not guarantee acceptable quantization accuracy. |
| Export destination already exists | The tested notebook may reuse it when overwrite is disabled. After checkpoint selection changes, use a fresh export name or back up the old folder before explicitly enabling overwrite. Inspect worker exit codes as well as saved manifests. |

## Reporting

The [v2 report](../results/v2/REPORT.md) contains completed NVFP4 quality and speed results. NVFP4 was faster than BF16 for E2B/E4B at high tested concurrency, with a model-dependent accuracy cost; 270M did not gain useful speed in this workload.

Retain the distinction between native W4A4 execution, calibration quality, and whole-model precision. Excluded components remain higher precision. Results do not establish that NVFP4 is universally faster or slower.
