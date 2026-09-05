# NVFP4 Serving Notes

## Pinned environment

The vLLM notebook requires this serving combination for native Blackwell W4A4
NVFP4 testing:

- GPU: NVIDIA Blackwell with compute capability 12.0 or newer.
- PyTorch: 2.13.0 with CUDA 13.2.
- vLLM: 0.27.0.
- Transformers: 5.14.1.
- CUDA JIT packages: all of `nvidia-cuda-cccl`, `nvidia-cuda-crt`,
  `nvidia-cuda-nvcc`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`,
  `nvidia-nvjitlink`, and `nvidia-nvvm` at 13.2.86.

Cell 5 now reinstalls and verifies this full component set before any native
NVFP4 model loads. It also makes unversioned CUDA library links available to
the FlashInfer JIT linker.

## Expected native path

The notebook requests `flashinfer_cutlass` for dense NVFP4 linear layers and
`flashinfer_b12x` for Gemma 4 MoE experts. This is W4A4 NVFP4, not the
weight-only W4A16 fallback. The exported benchmark manifest must report:

- `quantization_kind: nvfp4`
- `native_nvfp4: true`
- `native_nvfp4_w4a4: true`
- `cuda13_toolkit_verified: true`

## Failure guide

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `CUDA compiler and CUDA toolkit headers are incompatible` | Mixed CUDA pip wheels, commonly 13.3 `nvcc` with 13.2 headers. | Restart the Colab runtime and rerun Cells 1-5 from the updated notebook. |
| `no 'flashinfer_b12x' kernel exists for NVFP4 layers` | B12x was forced for dense linear layers. | Keep dense backend `flashinfer_cutlass`; reserve B12x for MoE. |
| `cannot find -lcudart` or `-lnvrtc` | The JIT linker cannot resolve unversioned CUDA libraries. | Use the updated Cell 5, which creates a curated `LIBRARY_PATH`. |
| Native kernel error after model load | A vLLM, FlashInfer, or CUDA 13 Blackwell kernel issue. | Preserve the manifest, `flashinfer show-config`, GPU details, and exact error before trying a newer pinned stack. |
| Out of memory during vLLM startup | KV cache reservation is too large for the model and requested context/concurrency. | Reduce `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, or `VLLM_MAX_NUM_SEQS`. |
| Correct-looking startup but poor outputs | A native FP4 kernel correctness or quantization regression. | Compare greedy outputs with BF16 on the identical held-out set before reporting throughput. |

## Reporting policy

Treat BF16 and FP8 benchmark results as reportable after their held-out
accuracy and speed files are saved. NVFP4 remains experimental until one
native W4A4 benchmark completes and passes the same fixed-set quality checks.
