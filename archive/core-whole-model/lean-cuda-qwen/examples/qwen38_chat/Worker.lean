/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Lean.Data.Json
import LeanCudaQwen.Foundation
import LeanCudaQwen.Qwen36.Megakernel

/-!
# Persistent Qwen3.8 chat worker

Loads one checkpoint and accepts bounded generation requests over a tagged line protocol. Only
tagged protocol records are consumed by the HTTP bridge, so checkpoint progress remains readable.
-/

namespace Qwen38Chat
private structure BenchmarkConfig where
  warmup : Nat
  iters : Nat
  repeats : Nat
  runId : String

private def percentile (samples : Array Float) (pct : Nat) : Float :=
  if samples.isEmpty then 0.0 else
    let sorted := samples.qsort (· < ·)
    sorted[((sorted.size - 1) * pct) / 100]!

private structure BenchmarkSummary where
  event : String := "summary"
  schemaVersion : Nat := 1
  runId : String
  caseId : String
  backend : String
  routeActual : String
  timer : String
  completionFence : String
  timingScope : String
  warmAllocationFree : Bool
  warmup : Nat
  iters : Nat
  repeats : Nat
  correctnessOk : Bool
  latencyMsP10 : Float
  latencyMsP50 : Float
  latencyMsP90 : Float
  workItemsPerIteration : Option Float := none
  workItemUnit : Option String := none
  throughputItemsPerSecondP10 : Option Float := none
  throughputItemsPerSecondP50 : Option Float := none
  throughputItemsPerSecondP90 : Option Float := none
  deriving Lean.ToJson

private def benchmarkSummaryJson (config : BenchmarkConfig) (caseId backend route : String)
    (samples : Array Float) (correctnessOk : Bool) (timingScope : String)
    (warmAllocationFree : Bool) (timer completionFence : String)
    (workItemsPerIteration : Option Float := none) (workItemUnit : Option String := none) : String :=
  let latencyMsP10 := percentile samples 10
  let latencyMsP50 := percentile samples 50
  let latencyMsP90 := percentile samples 90
  let throughputAt (latencyMs : Float) := workItemsPerIteration.map fun workItems =>
    if latencyMs > 0.0 then workItems * 1000.0 / latencyMs else 0.0
  Lean.toJson ({
    runId := config.runId, caseId, backend, routeActual := route, timer, completionFence,
    timingScope, warmAllocationFree, warmup := config.warmup, iters := config.iters,
    repeats := config.repeats, correctnessOk, latencyMsP10, latencyMsP50, latencyMsP90,
    workItemsPerIteration, workItemUnit,
    throughputItemsPerSecondP10 := throughputAt latencyMsP90,
    throughputItemsPerSecondP50 := throughputAt latencyMsP50,
    throughputItemsPerSecondP90 := throughputAt latencyMsP10
  } : BenchmarkSummary) |>.compress


private structure GenerateRequest where
  maxNewTokens : Nat
  stopToken : UInt32
  cachePrefixTokens : Nat
  prompt : Array UInt32

private structure BenchmarkRequest where
  warmup : Nat
  repeats : Nat
  maxNewTokens : Nat
  prompt : Array UInt32

private inductive Request where
  | generate (request : GenerateRequest)
  | benchmark (request : BenchmarkRequest)

private def parseToken (name raw : String) : Except String UInt32 := do
  let some value := raw.toNat?
    | throw s!"{name} must be an unsigned integer"
  unless value < 248320 do
    throw s!"{name} must be below 248320"
  return value.toUInt32

private def parsePositiveNat (name raw : String) : Except String Nat := do
  let some value := raw.toNat?
    | throw s!"{name} must be an unsigned integer"
  unless value > 0 do
    throw s!"{name} must be positive"
  return value

private def parsePrompt (promptRaw : String) : Except String (Array UInt32) := do
  let mut prompt := #[]
  for raw in promptRaw.splitOn "," do
    prompt := prompt.push (← parseToken "prompt token" raw)
  unless !prompt.isEmpty do
    throw "prompt must contain at least one token"
  return prompt

private def parseRequest (line : String) : Except String Request := do
  match line.splitOn " " with
  | ["GENERATE", maxRaw, stopRaw, cachePrefixRaw, promptRaw] =>
    let maxNewTokens ← parsePositiveNat "max-new-tokens" maxRaw
    let stopToken ← parseToken "stop-token" stopRaw
    let cachePrefixTokens ← parsePositiveNat "cache-prefix-tokens" cachePrefixRaw
    let prompt ← parsePrompt promptRaw
    unless cachePrefixTokens < prompt.size do
      throw "cache-prefix-tokens must end before the assistant generation suffix"
    return .generate { maxNewTokens, stopToken, cachePrefixTokens, prompt }
  | ["BENCHMARK", warmupRaw, repeatsRaw, maxRaw, promptRaw] =>
    let some warmup := warmupRaw.toNat?
      | throw "warmup must be an unsigned integer"
    let repeats ← parsePositiveNat "repeats" repeatsRaw
    let maxNewTokens ← parsePositiveNat "max-new-tokens" maxRaw
    unless maxNewTokens > 1 do
      throw "benchmark max-new-tokens must be at least two"
    let prompt ← parsePrompt promptRaw
    return .benchmark { warmup, repeats, maxNewTokens, prompt }
  | command :: _ => throw s!"unknown command {command}"
  | [] => throw "empty request"

private def sanitizeError (error : IO.Error) : String :=
  ((toString error).replace "\r" " ").replace "\n" " "

private def send (stdout : IO.FS.Stream) (message : String) : IO Unit := do
  stdout.putStrLn ("QWEN_CHAT " ++ message)
  stdout.flush

private def generate (session : @& Cuda.Qwen36.Megakernel.GenerationSession)
    (stdout : IO.FS.Stream) (request : GenerateRequest) : IO Unit := do
  let generated ← session.generateStreamingWith request.prompt request.cachePrefixTokens {
      maxNewTokens := request.maxNewTokens
      stopTokens := #[request.stopToken]
    } fun event => send stdout s!"TOKEN {event.token}"
  send stdout s!"DONE {generated.size}"

private def elapsedMilliseconds (start finish : Nat) : Float :=
  (finish - start).toFloat / 1000000.0

private partial def firstTokenMismatch? (reference candidate : Array UInt32)
    (position : Nat := 0) : Option (Nat × Option UInt32 × Option UInt32) :=
  if position < reference.size || position < candidate.size then
    let referenceToken := reference[position]?
    let candidateToken := candidate[position]?
    if referenceToken == candidateToken then
      firstTokenMismatch? reference candidate (position + 1)
    else
      some (position, referenceToken, candidateToken)
  else
    none

private def renderToken? : Option UInt32 → String
  | some token => toString token
  | none => "<missing>"

private def benchmark (checkpoint : @& Cuda.Qwen36.Megakernel.Checkpoint)
    (stdout : IO.FS.Stream) (request : BenchmarkRequest) : IO Unit := do
  let generationConfig : Cuda.Qwen36.Megakernel.GenerationConfig := {
    maxNewTokens := request.maxNewTokens
    stopTokens := #[]
  }
  let expectedTokens ← checkpoint.generateStreamingReferenceWith request.prompt
    generationConfig fun _ => pure ()
  for _ in [:request.warmup] do
    let _ ← checkpoint.generateStreamingWith request.prompt generationConfig fun _ => pure ()
  let ttftSamples ← IO.mkRef (#[] : Array Float)
  let decodeSamples ← IO.mkRef (#[] : Array Float)
  let correctness ← IO.mkRef (expectedTokens.size == request.maxNewTokens)
  for iteration in [:request.repeats] do
    let start ← IO.monoNanosNow
    let firstTokenAt ← IO.mkRef (none : Option Nat)
    let finalTokenAt ← IO.mkRef start
    let generated ← checkpoint.generateStreamingWith request.prompt generationConfig fun _ => do
      let finish ← IO.monoNanosNow
      match ← firstTokenAt.get with
      | none =>
        ttftSamples.modify (·.push (elapsedMilliseconds start finish))
        firstTokenAt.set (some finish)
      | some _ => finalTokenAt.set finish
    if generated.size != request.maxNewTokens then
      correctness.set false
    if generated.size > 1 then
      match ← firstTokenAt.get with
      | none => correctness.set false
      | some first =>
        let finish ← finalTokenAt.get
        let meanDecodeMs :=
          elapsedMilliseconds first finish / (generated.size - 1).toFloat
        decodeSamples.modify (·.push meanDecodeMs)
    if let some (position, referenceToken, candidateToken) :=
        firstTokenMismatch? expectedTokens generated then
      correctness.set false
      send stdout <| s!"ERROR BF16/MXFP8 token mismatch repeat={iteration} position={position} " ++
        s!"reference={renderToken? referenceToken} candidate={renderToken? candidateToken}"
      return
  let ttft ← ttftSamples.get
  let decode ← decodeSamples.get
  let correct := (← correctness.get) &&
    ttft.size == request.repeats &&
    decode.size == request.repeats
  let runId := s!"qwen38_{← IO.monoNanosNow}"
  let ttftConfig : BenchmarkConfig := {
    warmup := request.warmup
    iters := 1
    repeats := request.repeats
    runId
  }
  let decodeConfig : BenchmarkConfig := {
    ttftConfig with iters := request.maxNewTokens - 1
  }
  let ttftJson := benchmarkSummaryJson ttftConfig "qwen38_27b_ttft" "lean_cuda"
    Cuda.Qwen36.Megakernel.benchmarkRoute ttft correct "end_to_end_first_token" false
    "IO.monoNanosNow" "token callback publication"
  let decodeJson := benchmarkSummaryJson decodeConfig "qwen38_27b_decode" "lean_cuda"
    Cuda.Qwen36.Megakernel.benchmarkRoute decode correct "steady_decode_callback_interval" true
    "IO.monoNanosNow" "token callback publication" (some 1.0) (some "token")
  send stdout s!"SUMMARY {ttftJson}"
  send stdout s!"SUMMARY {decodeJson}"
  send stdout s!"DONE {request.repeats}"

private partial def requestLoop (serving : @& Cuda.Qwen36.Megakernel.ServingCheckpoint)
    (stdin stdout : IO.FS.Stream) : IO Unit := do
  let raw ← stdin.getLine
  if raw.isEmpty then
    return
  let line := raw.trimAscii.copy
  if line == "QUIT" then
    send stdout "BYE"
    return
  match parseRequest line with
  | .error message => send stdout s!"ERROR {message}"
  | .ok (.generate request) =>
      try
        generate serving.generation stdout request
      catch error =>
        send stdout s!"ERROR {sanitizeError error}"
  | .ok (.benchmark request) =>
      try
        benchmark serving.checkpoint stdout request
      catch error =>
        send stdout s!"ERROR {sanitizeError error}"
  requestLoop serving stdin stdout

def run : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let some modelDirectory ← IO.getEnv "QWEN_MODEL_DIR"
    | send stdout "ERROR QWEN_MODEL_DIR is required"
      return 2
  try
    let serving ← Cuda.Qwen36.Megakernel.loadServingCheckpoint modelDirectory
    send stdout "READY BF16_EXACT_V1"
    requestLoop serving stdin stdout
    return 0
  catch error =>
    send stdout s!"ERROR {sanitizeError error}"
    return 1

end Qwen38Chat

public def main : IO UInt32 :=
  Qwen38Chat.run
