/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public meta import Lean.Elab.Command
public meta import LeanCudaQwen.Qwen36.Config
meta import Lean.Data.Json
meta import Lean.Parser.Extension

public section

/-!
# HuggingFace config.json type provider

`hfconfig_type_provider "path/to/config.json" as NS` reads a Qwen3.6 HF config at elaboration
time, validates `text_config` with `Config.check`, and emits into namespace `NS`:

- `config : Config` — the validated configuration value;
- one constant per scalar field (`hiddenSize`, `numHiddenLayers`, `linearNumKeyHeads`,
  `linearKeyHeadDim`, `linearNumValueHeads`, `linearValueHeadDim`, `linearConvKernelDim`,
  `numAttentionHeads`, `numKeyValueHeads`, `headDim`, `partialRotaryFactor`, `intermediateSize`,
  `vocabSize`, `rmsNormEps`, `ropeTheta`, `tieWordEmbeddings`);
- derived dimensions (`keyDim`, `valueDim`, `convDim`, `attnQDim`, `qProjDim`, `kvProjDim`,
  `gqaRatio`, `linearRepeatFactor`, `rotaryDim`) as literals, so they reduce by `rfl`;
- `layerTypes : Array LayerType` — the per-layer mixer table;
- `isFullAttention (layer : Nat) : Bool`.

`rope_theta` and `partial_rotary_factor` are read from `rope_parameters` when absent at the
`text_config` level (the 27B HF config nests them there). A JSON object without a `text_config`
member is treated as a bare text config. Elaboration fails on missing or malformed fields and on
any `Config.check` violation. Float constants are re-emitted through `OfScientific.ofScientific`
from the JSON mantissa/exponent, preserving the exact decimal value.

Pattern follows `Tyr/SafeTensors/TypeProvider.lean`: generate command source text, re-parse it
with `Parser.runParserCategory`, and elaborate each declaration into the target namespace.
-/

namespace Lean.Cuda.Qwen36

open Lean Elab Command

syntax (name := hfConfigTypeProvider) "hfconfig_type_provider " str " as " ident : command

/-- Exact decimal rendering of a JSON number as a Float term (`mantissa * 10^-exponent`). -/
meta def renderFloatLit (n : JsonNumber) : String :=
  let core := s!"(_root_.OfScientific.ofScientific {n.mantissa.natAbs} true {n.exponent} : _root_.Float)"
  if n.mantissa < 0 then s!"(-{core})" else core

meta def renderLayerType : LayerType → String
  | .linearAttention => ".linearAttention"
  | .fullAttention => ".fullAttention"

meta def elabGenerated (source : String) : CommandElabM Unit := do
  let env ← getEnv
  match Parser.runParserCategory env `command source "<hfconfig_type_provider>" with
  | .ok stx => elabCommand stx
  | .error err =>
    throwError "hfconfig_type_provider generated invalid Lean command:\n{err}\n\n{source}"

meta def lookupNat (obj : Json) (path name : String) : CommandElabM Nat := do
  match obj.getObjVal? name with
  | .ok v =>
    match v.getNat? with
    | .ok n => pure n
    | .error err =>
      throwError "hfconfig_type_provider: field '{name}' in '{path}' must be a natural number: {err}"
  | .error _ =>
    throwError "hfconfig_type_provider: missing field '{name}' in '{path}'"

meta def lookupBool (obj : Json) (path name : String) : CommandElabM Bool := do
  match obj.getObjVal? name with
  | .ok v =>
    match v.getBool? with
    | .ok b => pure b
    | .error err =>
      throwError "hfconfig_type_provider: field '{name}' in '{path}' must be a boolean: {err}"
  | .error _ =>
    throwError "hfconfig_type_provider: missing field '{name}' in '{path}'"

meta def asNumber (v : Json) (path name : String) : CommandElabM JsonNumber := do
  match v.getNum? with
  | .ok n => pure n
  | .error err =>
    throwError "hfconfig_type_provider: field '{name}' in '{path}' must be a number: {err}"

/-- Read `name` from the object, falling back to `rope_parameters.name` (HF nests RoPE settings). -/
meta def lookupNumber (obj : Json) (path name : String) : CommandElabM JsonNumber := do
  match obj.getObjVal? name with
  | .ok v => asNumber v path name
  | .error _ =>
    match obj.getObjVal? "rope_parameters" with
    | .ok rp =>
      match rp.getObjVal? name with
      | .ok v => asNumber v path s!"rope_parameters.{name}"
      | .error _ =>
        throwError "hfconfig_type_provider: missing field '{name}' (or 'rope_parameters.{name}') in '{path}'"
    | .error _ =>
      throwError "hfconfig_type_provider: missing field '{name}' (or 'rope_parameters.{name}') in '{path}'"

meta def lookupLayerTypes (obj : Json) (path : String) : CommandElabM (Array LayerType) := do
  let arr ←
    match obj.getObjVal? "layer_types" with
    | .ok v =>
      match v.getArr? with
      | .ok arr => pure arr
      | .error err =>
        throwError "hfconfig_type_provider: field 'layer_types' in '{path}' must be an array: {err}"
    | .error _ =>
      throwError "hfconfig_type_provider: missing field 'layer_types' in '{path}'"
  let mut layerTypes := #[]
  for v in arr do
    match v.getStr? with
    | .ok "linear_attention" => layerTypes := layerTypes.push .linearAttention
    | .ok "full_attention" => layerTypes := layerTypes.push .fullAttention
    | .ok other =>
      throwError "hfconfig_type_provider: unknown layer type '{other}' in '{path}' (expected 'linear_attention' or 'full_attention')"
    | .error err =>
      throwError "hfconfig_type_provider: 'layer_types' entries in '{path}' must be strings: {err}"
  pure layerTypes

meta def elabHfConfigTypeProviderCore (path : String) (ns : Name) : CommandElabM Unit := do
  let raw ←
    try liftIO (IO.FS.readFile ⟨path⟩)
    catch e => throwError "hfconfig_type_provider: failed to read '{path}': {e.toMessageData}"
  let json ←
    match Json.parse raw with
    | .ok j => pure j
    | .error err => throwError "hfconfig_type_provider: failed to parse '{path}' as JSON: {err}"
  let text ←
    match json.getObjVal? "text_config" with
    | .ok tc => pure tc
    | .error _ => pure json
  let hiddenSize ← lookupNat text path "hidden_size"
  let numHiddenLayers ← lookupNat text path "num_hidden_layers"
  let layerTypes ← lookupLayerTypes text path
  let linearNumKeyHeads ← lookupNat text path "linear_num_key_heads"
  let linearKeyHeadDim ← lookupNat text path "linear_key_head_dim"
  let linearNumValueHeads ← lookupNat text path "linear_num_value_heads"
  let linearValueHeadDim ← lookupNat text path "linear_value_head_dim"
  let linearConvKernelDim ← lookupNat text path "linear_conv_kernel_dim"
  let numAttentionHeads ← lookupNat text path "num_attention_heads"
  let numKeyValueHeads ← lookupNat text path "num_key_value_heads"
  let headDim ← lookupNat text path "head_dim"
  let intermediateSize ← lookupNat text path "intermediate_size"
  let vocabSize ← lookupNat text path "vocab_size"
  let tieWordEmbeddings ← lookupBool text path "tie_word_embeddings"
  -- JSON numbers, kept alongside the Float values for exact literal re-emission.
  let partialRotaryFactor ← lookupNumber text path "partial_rotary_factor"
  let rmsNormEps ← lookupNumber text path "rms_norm_eps"
  let ropeTheta ← lookupNumber text path "rope_theta"
  let cfg : Config := {
    hiddenSize, numHiddenLayers, layerTypes,
    linearNumKeyHeads, linearKeyHeadDim, linearNumValueHeads, linearValueHeadDim,
    linearConvKernelDim, numAttentionHeads, numKeyValueHeads, headDim,
    partialRotaryFactor := partialRotaryFactor.toFloat,
    intermediateSize, vocabSize,
    rmsNormEps := rmsNormEps.toFloat,
    ropeTheta := ropeTheta.toFloat,
    tieWordEmbeddings
  }
  let cfg ←
    match cfg.check with
    | .ok valid => pure valid
    | .error err => throwError "hfconfig_type_provider: '{path}': {err}"

  let qroot := "_root_.Lean.Cuda.Qwen36"
  let layerTypesLit :=
    "#[" ++ String.intercalate ", " (cfg.layerTypes.toList.map renderLayerType) ++ "]"
  let configLit := String.intercalate ", " [
    s!"hiddenSize := {cfg.hiddenSize}",
    s!"numHiddenLayers := {cfg.numHiddenLayers}",
    s!"layerTypes := {layerTypesLit}",
    s!"linearNumKeyHeads := {cfg.linearNumKeyHeads}",
    s!"linearKeyHeadDim := {cfg.linearKeyHeadDim}",
    s!"linearNumValueHeads := {cfg.linearNumValueHeads}",
    s!"linearValueHeadDim := {cfg.linearValueHeadDim}",
    s!"linearConvKernelDim := {cfg.linearConvKernelDim}",
    s!"numAttentionHeads := {cfg.numAttentionHeads}",
    s!"numKeyValueHeads := {cfg.numKeyValueHeads}",
    s!"headDim := {cfg.headDim}",
    s!"partialRotaryFactor := {renderFloatLit partialRotaryFactor}",
    s!"intermediateSize := {cfg.intermediateSize}",
    s!"vocabSize := {cfg.vocabSize}",
    s!"rmsNormEps := {renderFloatLit rmsNormEps}",
    s!"ropeTheta := {renderFloatLit ropeTheta}",
    s!"tieWordEmbeddings := {cfg.tieWordEmbeddings}" ]
  let natConsts : Array (String × Nat) := #[
    ("hiddenSize", cfg.hiddenSize),
    ("numHiddenLayers", cfg.numHiddenLayers),
    ("linearNumKeyHeads", cfg.linearNumKeyHeads),
    ("linearKeyHeadDim", cfg.linearKeyHeadDim),
    ("linearNumValueHeads", cfg.linearNumValueHeads),
    ("linearValueHeadDim", cfg.linearValueHeadDim),
    ("linearConvKernelDim", cfg.linearConvKernelDim),
    ("numAttentionHeads", cfg.numAttentionHeads),
    ("numKeyValueHeads", cfg.numKeyValueHeads),
    ("headDim", cfg.headDim),
    ("intermediateSize", cfg.intermediateSize),
    ("vocabSize", cfg.vocabSize),
    ("keyDim", cfg.keyDim),
    ("valueDim", cfg.valueDim),
    ("convDim", cfg.convDim),
    ("attnQDim", cfg.attnQDim),
    ("qProjDim", cfg.qProjDim),
    ("kvProjDim", cfg.kvProjDim),
    ("gqaRatio", cfg.gqaRatio),
    ("linearRepeatFactor", cfg.linearRepeatFactor),
    ("rotaryDim", cfg.rotaryDim)
  ]
  let nsName := ns.toString
  elabGenerated s!"namespace {nsName}"
  try
    elabGenerated <|
      s!"/-- Validated Qwen3.6 text configuration from `{path}`. -/\n" ++
      ("def config : " ++ qroot ++ ".Config := { " ++ configLit ++ " }")
    for (name, value) in natConsts do
      elabGenerated s!"def {name} : Nat := {value}"
    elabGenerated <|
      s!"/-- Per-layer mixer table from `{path}`. -/\n" ++
      s!"def layerTypes : Array {qroot}.LayerType := {layerTypesLit}"
    elabGenerated s!"def partialRotaryFactor : Float := {renderFloatLit partialRotaryFactor}"
    elabGenerated s!"def rmsNormEps : Float := {renderFloatLit rmsNormEps}"
    elabGenerated s!"def ropeTheta : Float := {renderFloatLit ropeTheta}"
    elabGenerated s!"def tieWordEmbeddings : Bool := {cfg.tieWordEmbeddings}"
    elabGenerated <|
      s!"/-- Whether `layer` is a gated full-attention layer. -/\n" ++
      s!"def isFullAttention (layer : Nat) : Bool := config.isFullAttention layer"
  finally
    elabGenerated s!"end {nsName}"

@[command_elab hfConfigTypeProvider]
meta def elabHfConfigTypeProvider : CommandElab
  | `(hfconfig_type_provider $path:str as $ns:ident) =>
      elabHfConfigTypeProviderCore path.getString ns.getId
  | _ => throwUnsupportedSyntax

end Lean.Cuda.Qwen36
