/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Config
public meta import LeanCudaQwen.Qwen36.ConfigProvider
public import LeanCudaQwen.Qwen36.Primitives
public import LeanCudaQwen.Qwen36.Linear
public import LeanCudaQwen.Qwen36.LoRA
public import LeanCudaQwen.Qwen36.Projection
public import LeanCudaQwen.Qwen36.DeltaNet
public import LeanCudaQwen.Qwen36.Attention
public import LeanCudaQwen.Qwen36.MLP
public import LeanCudaQwen.Qwen36.Model
public import LeanCudaQwen.Qwen36.Training
public import LeanCudaQwen.Qwen36.PretrainMegakernel
public import LeanCudaQwen.Qwen36.GRPOMegakernel
public import LeanCudaQwen.Qwen36.FullTrainingMegakernelCore
public import LeanCudaQwen.Qwen36.FullPretrainMegakernel
public import LeanCudaQwen.Qwen36.FullGRPOMegakernel
public import LeanCudaQwen.Qwen36.SafeTensors
public import LeanCudaQwen.Qwen36.Megakernel
public import LeanCudaQwen.Qwen36.CheckpointLoRA

public section

/-!
# Qwen3.6-27B megakernel training and inference

Umbrella for the Qwen3.6 text-model implementation (see `docs/QWEN36_MEGAKERNEL.md`). It exposes
the typed HF configuration provider, shared forward/backward CUDA primitives, mixed-batch LoRA,
and native safetensors loading, the complete sequential numerical model, bounded persistent
multi-token real-checkpoint inference, and the real-checkpoint LoRA bridge. The resident training
megakernel joins this surface with its parity gate. The real-checkpoint bridge currently trains a
mixed adapter bank on the LM head; native base-model batching and projection-wide checkpoint LoRA
remain distinct production gates.
-/
