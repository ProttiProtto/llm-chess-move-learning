# Precision and performance guide

**Scope:** the published v2 experiments trained all six runs with BF16 base weights, then merged and post-training-quantized with the vLLM export notebook (compressed-tensors FP8_DYNAMIC and ModelOpt NVFP4 W4A4). The TorchAO FP8/QAT paths described below are optional implementations, not the source of the reported quantization results. Their convergence, speed, and compatibility with ModelOpt exports have not been established by this experiment matrix. See [the report](../results/v2/REPORT.md).

## FP8 training versus export

The `fp8` training mode and FP8 deployment artifact are intentionally separate:

- Training uses TorchAO `Float8Linear` to cast eligible GEMM operands to FP8 in forward and backward while keeping BF16 master weights. LoRA parameters and optimizer state remain higher precision.
- Export first merges LoRA into a BF16 copy, then applies persistent FP8 activation-and-weight quantization and saves the FP8 values and scales through the Transformers TorchAO integration.
- A saved adapter is still a LoRA checkpoint. A successfully exported `fp8_dynamic_activation_weight` directory is a merged FP8 model and no longer requires PEFT at inference time.

This should be described as **FP8 mixed-precision LoRA with merged FP8 inference export**, not conventional four-bit QLoRA.

This pipeline has three precision modes. They all keep the trainable LoRA
parameters and optimizer state in higher precision. The mode primarily changes
the matrix multiplications inside eligible base-model linear layers.

## BF16

BF16 uses one sign bit, eight exponent bits, and seven explicit fraction bits.
Its exponent range is close to FP32, so it is substantially easier to train
than FP16 without loss scaling. Model weights and activations are loaded in
BF16, while optimizers normally keep FP32 state.

On Ampere and newer NVIDIA GPUs, BF16 matrix multiplications use Tensor Cores.
It is the compatibility and numerical baseline. It uses two bytes per stored
weight or activation and does not pay quantize/dequantize overhead.

## TorchAO FP8 training

`precision.mode: fp8` replaces eligible `torch.nn.Linear` modules with
TorchAO `Float8Linear` modules. For a linear operation, the BF16 input and
weight are scaled and cast to FP8, the Tensor Core performs the FP8 matrix
multiplication, and its result is accumulated/returned in higher precision.
The same pattern is used for the two backward matrix multiplications.

The configured `rowwise` recipe computes a separate scale for each matrix row.
This handles outliers better than one scale for an entire tensor, but scale
calculation and casts cost time. `tensorwise` usually has less overhead;
`rowwise_with_gw_hp` keeps the weight-gradient path at higher precision. These
recipes must be benchmarked on the exact model because small GEMMs can be
slower than BF16 once scaling overhead is included.

FP8 is a compute format here, not an eight-bit checkpoint format. BF16 master
weights, gradients where required, LoRA parameters, and optimizer state remain
available for updates. Peak training memory therefore does not automatically
halve. Eligible dimensions and padded sequence dimensions are multiples of 16
so the hardware kernels can be selected.

Hardware: CUDA compute capability 8.9 or newer for this implementation. L4,
H100/H200, Ada RTX, and Blackwell qualify; T4 and A100 do not provide this FP8
path. `torch.compile` is important because it fuses scaling/casting work around
the GEMMs.

## TorchAO NVFP4 QAT

`precision.mode: nvfp4_qat` uses `NVFP4InferenceConfig` with TorchAO's
recommended `QATConfig(..., step="prepare")` flow. This is genuinely the NVFP4
numerical format, not generic integer INT4. It is nevertheless fake-quantized
training: values are rounded onto the NVFP4 grid and then represented again in
higher precision for the training operations.

An NVFP4 value is E2M1: one sign bit, two exponent bits, and one mantissa bit.
Sixteen consecutive values share an FP8 E4M3 block scale, and the tensor also
has a global FP32 scale. Conceptually:

```text
reconstructed_value = E2M1_value * FP8_block_scale * FP32_global_scale
```

Fine-grained scaling gives NVFP4 more usable dynamic range than unscaled FP4.
QAT is intended to expose trainable updates to rounding error. Whether this
particular path recovers accuracy after a merged ModelOpt export needs a
separate, recipe-matched experiment; it is not demonstrated by the PTQ results.

The current TorchAO QAT path does not execute training GEMMs on native NVFP4
Tensor Cores, does not keep optimizer state in four bits, and should not be
selected as a training-speed or training-memory optimization. Its purpose is
to make the resulting adapter robust when a merged model is later converted to
NVFP4 for inference. It requires Blackwell and remains a prototype API.

## Native NVFP4 training

NVIDIA Transformer Engine has a native `NVFP4BlockScaling` training recipe for
Blackwell. It uses E2M1 values, block/global scaling, stochastic rounding,
random Hadamard transforms, and 2D weight scaling, then runs supported GEMMs on
Blackwell Tensor Cores. That is the path intended for actual FP4 training
throughput.

Transformer Engine only accelerates its own modules. Converting arbitrary
Hugging Face Gemma modules while preserving PEFT LoRA wrapping, Gemma 4's
clippable linear wrappers, generation, and checkpoint compatibility requires a
model-specific integration. It is therefore not silently substituted for the
portable TorchAO QAT path in this pipeline. A future native mode should be
implemented and validated separately against BF16 convergence before use in a
reported experiment.

## `torch.compile` and speed decisions

Inductor compilation traces the forward and backward graphs, fuses compatible
operations, and generates Triton/CUDA kernels. The first step is slow because
it compiles; steady-state steps should be compared only after warm-up.

`torch_compile_strategy: auto` resolves to `full` for Gemma 3 and to
`regional` for Gemma 4. Gemma 4 combines per-layer embeddings, shared KV state,
optional MoE branches, and LoRA in a very large training graph. With PyTorch
2.11, full-model AOTAutograd partitioning can fail before Inductor emits a
kernel. Regional mode compiles each repeated text decoder layer's forward
method instead, which avoids the monolithic partition while preserving module
and checkpoint names. Explicit `full` and `regional` modes remain available
for controlled benchmarks.

- `default` is the safest mode for variable sequence lengths.
- Regional mode uses dynamic shapes by default to avoid recompiling for every
  batch-specific padded sequence length.
- `reduce-overhead` uses CUDA graphs and may consume more memory or recompile
  when shapes change.
- `max-autotune-no-cudagraphs` spends longer benchmarking kernels and is useful
  for long runs with stable shapes.
- A larger per-device microbatch normally improves GPU utilization. Reduce
  gradient accumulation to keep the same effective batch for fair experiments.
- Gradient checkpointing recomputes activations during backward. Disable it
  when VRAM permits.
- Periodic autoregressive validation can dominate wall time. Measure training
  throughput separately and choose validation frequency explicitly.
- `adamw_torch_fused`, TF32 for remaining FP32 operations, pinned-memory
  loading, and persistent workers are enabled as independent controls.

Primary references:

- https://docs.pytorch.org/ao/stable/workflows/training.html
- https://docs.pytorch.org/ao/stable/workflows/qat.html
- https://huggingface.co/docs/transformers/torch_compile
- https://docs.pytorch.org/tutorials/recipes/regional_compilation.html
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html
