/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Lean

namespace Qwen36.Checkpoint

open Lean

/-- Location and shape metadata for one tensor inside a safetensors data section. -/
structure TensorInfo where
  dtype : String
  shape : Array Nat
  dataStart : Nat
  dataEnd : Nat
  deriving Repr, BEq

/-- Parsed safetensors header. Offsets in `TensorInfo` are relative to `dataOffset`. -/
structure Header where
  headerSize : Nat
  dataOffset : Nat
  dataBytes : Nat
  tensors : Array (String × TensorInfo)
  deriving Repr

/-- Hugging Face sharded-checkpoint index. -/
structure WeightIndex where
  totalSize : Nat
  weights : Array (String × String)
  deriving Repr

/-- One validated checkpoint shard. -/
structure Shard where
  filename : String
  header : Header
  deriving Repr

/-- A validated sharded safetensors checkpoint. -/
structure Manifest where
  root : System.FilePath
  index : WeightIndex
  shards : Array Shard
  deriving Repr

private def jsonNat (context : String) : Json → Except String Nat
  | .num number =>
    let scale : Int := Int.ofNat (10 ^ number.exponent)
    if number.mantissa < 0 then
      throw s!"{context}: expected a nonnegative integer"
    else if number.mantissa % scale != 0 then
      throw s!"{context}: expected an integer, got {number}"
    else
      return (number.mantissa / scale).toNat
  | _ => throw s!"{context}: expected a JSON number"

private def jsonString (context : String) (json : Json) : Except String String := do
  match json with
  | .str value => return value
  | _ => throw s!"{context}: expected a JSON string"

private def objectField (context field : String) (json : Json) : Except String Json := do
  match json.getObjVal? field with
  | .ok value => return value
  | .error _ => throw s!"{context}: missing property '{field}'"

private def dtypeByteWidth : String → Option Nat
  | "BOOL" | "I8" | "U8" => some 1
  | "BF16" | "F16" | "I16" | "U16" => some 2
  | "F32" | "I32" | "U32" => some 4
  | "F64" | "I64" | "U64" => some 8
  | _ => none

private def parseTensorInfo (name : String) (json : Json) : Except String TensorInfo := do
  let dtype ← jsonString s!"tensor '{name}' dtype" (← objectField s!"tensor '{name}'" "dtype" json)
  let some byteWidth := dtypeByteWidth dtype
    | throw s!"tensor '{name}': unsupported dtype '{dtype}'"
  let shapeJson ← objectField s!"tensor '{name}'" "shape" json
  let shapeValues ← match shapeJson.getArr? with
    | .ok values => pure values
    | .error _ => throw s!"tensor '{name}' shape: expected an array"
  let mut shape := #[]
  for dimension in shapeValues do
    shape := shape.push (← jsonNat s!"tensor '{name}' shape" dimension)
  let offsetsJson ← objectField s!"tensor '{name}'" "data_offsets" json
  let offsets ← match offsetsJson.getArr? with
    | .ok values => pure values
    | .error _ => throw s!"tensor '{name}' data_offsets: expected an array"
  let (startJson, endJson) ← match offsets.toList with
    | [startJson, endJson] => pure (startJson, endJson)
    | _ => throw s!"tensor '{name}' data_offsets: expected exactly two offsets"
  let dataStart ← jsonNat s!"tensor '{name}' start offset" startJson
  let dataEnd ← jsonNat s!"tensor '{name}' end offset" endJson
  if dataEnd < dataStart then
    throw s!"tensor '{name}': end offset precedes start offset"
  let elementCount := shape.foldl (init := 1) (· * ·)
  let expectedBytes := elementCount * byteWidth
  if dataEnd - dataStart != expectedBytes then
    throw s!"tensor '{name}': shape and dtype require {expectedBytes} bytes, header has {dataEnd - dataStart}"
  return { dtype, shape, dataStart, dataEnd }

/-- Parse and validate the JSON portion of a safetensors header. -/
def parseHeaderJson (headerSize : Nat) (json : Json) : Except String Header := do
  let object ← match json.getObj? with
    | .ok object => pure object
    | .error _ => throw "safetensors header: expected a JSON object"
  let mut tensors := #[]
  for (name, value) in object.toList do
    if name != "__metadata__" then
      tensors := tensors.push (name, ← parseTensorInfo name value)
  let sorted := tensors.qsort fun left right => left.2.dataStart < right.2.dataStart
  let mut cursor := 0
  for (name, tensor) in sorted do
    if tensor.dataStart != cursor then
      throw s!"tensor '{name}': expected contiguous offset {cursor}, got {tensor.dataStart}"
    cursor := tensor.dataEnd
  return {
    headerSize
    dataOffset := 8 + headerSize
    dataBytes := cursor
    tensors
  }

/-- Parse a UTF-8 safetensors header without touching its tensor payload. -/
def parseHeaderText (headerSize : Nat) (text : String) : Except String Header := do
  let json ← Json.parse text
  parseHeaderJson headerSize json

private def decodeUInt64LE (bytes : ByteArray) : Except String Nat := do
  if bytes.size != 8 then
    throw s!"safetensors prefix: expected 8 bytes, got {bytes.size}"
  let mut value := 0
  let mut scale := 1
  for index in [0:8] do
    value := value + bytes[index]!.toNat * scale
    scale := scale * 256
  return value

private partial def readExactLoop
    (handle : IO.FS.Handle) (remaining : Nat) (buffer : ByteArray) : IO ByteArray := do
  if remaining == 0 then
    return buffer
  let request := min remaining (1024 * 1024)
  let chunk ← handle.read request.toUSize
  if chunk.isEmpty then
    throw <| IO.userError s!"unexpected end of file with {remaining} bytes left to read"
  readExactLoop handle (remaining - chunk.size) (buffer ++ chunk)

private def readExact (handle : IO.FS.Handle) (count : Nat) : IO ByteArray :=
  readExactLoop handle count .empty

private partial def discardExact (handle : IO.FS.Handle) (remaining : Nat) : IO Unit := do
  if remaining == 0 then
    return ()
  let request := min remaining (1024 * 1024)
  let chunk ← handle.read request.toUSize
  if chunk.isEmpty then
    throw <| IO.userError s!"unexpected end of file while skipping {remaining} bytes"
  discardExact handle (remaining - chunk.size)

/-- Read and fully validate only the header of a safetensors file. -/
def readHeader (path : System.FilePath) : IO Header := do
  let handle ← IO.FS.Handle.mk path .read
  let prefixBytes ← readExact handle 8
  let headerSize ← IO.ofExcept <| decodeUInt64LE prefixBytes
  if headerSize > 16 * 1024 * 1024 then
    throw <| IO.userError s!"{path}: implausible safetensors header size {headerSize}"
  let headerBytes ← readExact handle headerSize
  let some headerText := String.fromUTF8? headerBytes
    | throw <| IO.userError s!"{path}: safetensors header is not UTF-8"
  let header ← IO.ofExcept <| parseHeaderText headerSize headerText
  let metadata ← path.metadata
  let expectedFileSize := header.dataOffset + header.dataBytes
  if metadata.byteSize.toNat != expectedFileSize then
    throw <| IO.userError
      s!"{path}: expected file size {expectedFileSize}, got {metadata.byteSize.toNat}"
  return header

/--
Read the contiguous tensor-data section of a previously validated safetensors file.

This materializes the entire payload and is intended only for small fixtures and explicit host-side
inspection. Model-scale CUDA loading should use `forEachShardPayloadChunk` instead.
-/
def readShardPayload (path : System.FilePath) (header : Header) : IO ByteArray := do
  let handle ← IO.FS.Handle.mk path .read
  discardExact handle header.dataOffset
  readExact handle header.dataBytes

/-- Default upper bound for temporary host storage while uploading a checkpoint shard. -/
def defaultPayloadChunkBytes : Nat := 64 * 1024 * 1024

private partial def forEachPayloadChunkLoop
    (handle : IO.FS.Handle) (remaining offset chunkBytes : Nat)
    (consume : Nat → ByteArray → IO Unit) : IO Unit := do
  if remaining == 0 then
    return
  let request := min remaining chunkBytes
  let chunk ← handle.read request.toUSize
  if chunk.isEmpty then
    throw <| IO.userError s!"unexpected end of shard payload with {remaining} bytes left to read"
  consume offset chunk
  forEachPayloadChunkLoop handle (remaining - chunk.size) (offset + chunk.size) chunkBytes consume

/--
Consume a validated shard payload in bounded chunks without materializing it as a model-sized
`ByteArray`. Offsets are relative to the safetensors data section, so a consumer can copy each
chunk directly into the corresponding byte range of one CUDA allocation.
-/
def forEachShardPayloadChunk (path : System.FilePath) (header : Header)
    (consume : Nat → ByteArray → IO Unit)
    (chunkBytes : Nat := defaultPayloadChunkBytes) : IO Unit := do
  if chunkBytes == 0 then
    throw <| IO.userError "safetensors payload chunk size must be positive"
  IO.FS.withFile path .read fun handle => do
    discardExact handle header.dataOffset
    forEachPayloadChunkLoop handle header.dataBytes 0 chunkBytes consume

/-- Find a tensor in one parsed header. -/
def Header.findTensor? (header : Header) (name : String) : Option TensorInfo :=
  (header.tensors.find? fun entry => entry.1 == name).map (·.2)

/-- Parse a Hugging Face `model.safetensors.index.json`. -/
def readWeightIndex (path : System.FilePath) : IO WeightIndex := do
  let text ← IO.FS.readFile path
  let json ← IO.ofExcept <| Json.parse text
  let metadata ← IO.ofExcept <| objectField "checkpoint index" "metadata" json
  let totalSizeJson ← IO.ofExcept <| objectField "checkpoint index metadata" "total_size" metadata
  let totalSize ← IO.ofExcept <| jsonNat "checkpoint total_size" totalSizeJson
  let weightMapJson ← IO.ofExcept <| objectField "checkpoint index" "weight_map" json
  let weightMap : Std.TreeMap.Raw String Json compare ← IO.ofExcept weightMapJson.getObj?
  let mut weights := #[]
  for (name, shardJson) in weightMap.toList do
    let shard ← IO.ofExcept <| jsonString s!"checkpoint tensor '{name}'" shardJson
    weights := weights.push (name, shard)
  return { totalSize, weights }

private def uniqueShardNames (index : WeightIndex) : Array String := Id.run do
  let mut names := #[]
  for (_, filename) in index.weights do
    if !names.contains filename then
      names := names.push filename
  return names.qsort (· < ·)

/-- Resolve a tensor name to its shard according to the Hugging Face index. -/
def WeightIndex.findShard? (index : WeightIndex) (name : String) : Option String :=
  (index.weights.find? fun entry => entry.1 == name).map (·.2)

/-- Find a validated shard by filename. -/
def Manifest.findShard? (manifest : Manifest) (filename : String) : Option Shard :=
  manifest.shards.find? fun shard => shard.filename == filename

/-- Find a tensor and its shard in a validated manifest. -/
def Manifest.findTensor? (manifest : Manifest) (name : String) : Option (Shard × TensorInfo) := do
  let filename ← manifest.index.findShard? name
  let shard ← manifest.findShard? filename
  let tensor ← shard.header.findTensor? name
  return (shard, tensor)

/--
Read every shard header and prove that the index, headers, file lengths, and aggregate payload size
agree. Tensor payloads are not read.
-/
def readManifest (root : System.FilePath) : IO Manifest := do
  let root ← IO.FS.realPath root
  let index ← readWeightIndex (root / "model.safetensors.index.json")
  let mut shards := #[]
  for filename in uniqueShardNames index do
    let header ← readHeader (root / filename)
    shards := shards.push { filename, header }
  let manifest : Manifest := { root, index, shards }
  let headerTensorCount := shards.foldl (init := 0) fun count shard =>
    count + shard.header.tensors.size
  if headerTensorCount != index.weights.size then
    throw <| IO.userError
      s!"checkpoint index has {index.weights.size} tensors, shard headers have {headerTensorCount}"
  let payloadBytes := shards.foldl (init := 0) fun count shard =>
    count + shard.header.dataBytes
  if payloadBytes != index.totalSize then
    throw <| IO.userError
      s!"checkpoint index declares {index.totalSize} bytes, shard headers declare {payloadBytes}"
  for (name, filename) in index.weights do
    let some shard := manifest.findShard? filename
      | throw <| IO.userError s!"checkpoint tensor '{name}' references missing shard '{filename}'"
    if (shard.header.findTensor? name).isNone then
      throw <| IO.userError s!"checkpoint tensor '{name}' is absent from shard '{filename}'"
  for shard in shards do
    for (name, _) in shard.header.tensors do
      match index.findShard? name with
      | some filename =>
        if filename != shard.filename then
          throw <| IO.userError
            s!"checkpoint tensor '{name}' is in '{shard.filename}' but index names '{filename}'"
      | none =>
        throw <| IO.userError s!"shard '{shard.filename}' has unindexed tensor '{name}'"
  return manifest

/--
Stream one tensor from a validated sharded checkpoint. This deliberately avoids mapping or loading
the rest of a potentially multi-gigabyte shard.
-/
def readTensor (manifest : Manifest) (name : String) : IO (TensorInfo × ByteArray) := do
  let some (shard, tensor) := manifest.findTensor? name
    | throw <| IO.userError s!"checkpoint tensor not found: {name}"
  let handle ← IO.FS.Handle.mk (manifest.root / shard.filename) .read
  discardExact handle (shard.header.dataOffset + tensor.dataStart)
  let bytes ← readExact handle (tensor.dataEnd - tensor.dataStart)
  return (tensor, bytes)

end Qwen36.Checkpoint
