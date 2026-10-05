/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import Lean.Cuda.Types
import Lean.Data.Json
import Init.System.IO
import Init.Data.String.Legacy

public section

/-!
# SafeTensors loading for the Qwen3.8 CUDA backend

Lean-side reader for the safetensors checkpoint format: parses the 8-byte little-endian header
length plus JSON header (`dtype`, `shape`, `data_offsets`), introspects single-file checkpoints
and sharded directories via `model.safetensors.index.json`, and uploads raw tensor bytes to CUDA
device buffers. Loads never convert dtypes: a device buffer holds the exact file bytes and the
`TensorSchema` carries the element type.

The registry section maps the Qwen3.8-27B text-model HuggingFace tensor names — verified against
the published `model.safetensors.index.json`, which names `A_log` and `dt_bias` without a
`.weight` suffix — to typed weight families, and `checkComplete` validates a checkpoint schema
against the families a configuration expects. The vision tower (`model.visual.*`) and the MTP
layer (`mtp.*`) are outside text scope: they are reported separately and never count as missing
or unexpected.
-/

namespace Cuda.Qwen36.SafeTensors

/-- Safetensors header `dtype` tag. Loads perform no conversion; consumers read bytes per this. -/
inductive DType where
  | bool | u8 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64
  deriving Repr, BEq, DecidableEq, Inhabited

namespace DType

/-- Bytes per element in the data section. -/
def byteSize : DType → Nat
  | .bool | .u8 | .i8 => 1
  | .i16 | .f16 | .bf16 => 2
  | .i32 | .f32 => 4
  | .i64 | .f64 => 8

/-- The canonical header spelling. -/
def name : DType → String
  | .bool => "BOOL" | .u8 => "U8" | .i8 => "I8" | .i16 => "I16" | .i32 => "I32" | .i64 => "I64"
  | .f16 => "F16" | .bf16 => "BF16" | .f32 => "F32" | .f64 => "F64"

def ofName? : String → Option DType
  | "BOOL" => some .bool | "U8" => some .u8 | "I8" => some .i8 | "I16" => some .i16
  | "I32" => some .i32 | "I64" => some .i64 | "F16" => some .f16 | "BF16" => some .bf16
  | "F32" => some .f32 | "F64" => some .f64
  | _ => none

instance : ToString DType := ⟨DType.name⟩

end DType

/-- One tensor entry of a safetensors header, with the file offsets needed to load it. -/
structure TensorSchema where
  name : String
  dtype : DType
  shape : Array Nat
  /-- File offset where the owning shard's data section begins (`8 + header length`). -/
  dataOffset : Nat
  /-- Tensor byte range within the data section (header `data_offsets`). -/
  dataBegin : Nat
  dataEnd : Nat
  /-- Shard file relative to the source directory; empty for a single-file source. -/
  sourceFile : String := ""
  deriving Repr, Inhabited

namespace TensorSchema

/-- Number of elements. -/
def numel (t : TensorSchema) : Nat := t.shape.foldl (· * ·) 1

/-- Bytes implied by `shape × dtype`. -/
def byteSize (t : TensorSchema) : Nat := t.numel * t.dtype.byteSize

/-- Bytes spanned by `data_offsets`. -/
def span (t : TensorSchema) : Nat := t.dataEnd - t.dataBegin

/-- Header self-consistency: the byte span must match `shape × dtype`. -/
def check (t : TensorSchema) : Except String Unit :=
  if t.dataEnd < t.dataBegin then
    .error s!"tensor '{t.name}': data_offsets [{t.dataBegin}, {t.dataEnd}) are reversed"
  else if t.span != t.byteSize then
    .error s!"tensor '{t.name}': shape × dtype implies {t.byteSize} bytes, data_offsets span {t.span}"
  else
    .ok ()

end TensorSchema

private def readUInt64LE (bytes : ByteArray) (offset : Nat) : Option UInt64 :=
  if offset + 8 > bytes.size then none
  else Id.run do
    let mut n : UInt64 := 0
    for i in Array.range 8 do
      n := n ||| (bytes[offset + i]!.toUInt64 <<< (8 * i).toUInt64)
    return n

private def parseEntry (name sourceFile : String) (dataOffset : Nat) (entry : Lean.Json) :
    Except String TensorSchema := do
  let dtypeStr ← (← entry.getObjVal? "dtype").getStr?
  let dtype ← match DType.ofName? dtypeStr with
    | some d => .ok d
    | none => .error s!"tensor '{name}': unsupported dtype '{dtypeStr}'"
  let dims ← (← entry.getObjVal? "shape").getArr?
  let mut shape := #[]
  for d in dims do
    shape := shape.push (← d.getNat?)
  let offsets ← (← entry.getObjVal? "data_offsets").getArr?
  if offsets.size != 2 then
    .error s!"tensor '{name}': data_offsets must have exactly two entries"
  let dataBegin ← offsets[0]!.getNat?
  let dataEnd ← offsets[1]!.getNat?
  let schema := { name, dtype, shape, dataOffset, dataBegin, dataEnd, sourceFile : TensorSchema }
  schema.check
  return schema

/-- Parse the JSON header object of one safetensors file; `__metadata__` is skipped. -/
def parseHeaderObject (sourceFile : String) (dataOffset : Nat) (header : Lean.Json) :
    Except String (Array TensorSchema) := do
  let kvs ← header.getObj?
  let mut out := #[]
  for (name, entry) in kvs.toList do
    if name != "__metadata__" then
      out := out.push (← parseEntry name sourceFile dataOffset entry)
  return out

/-- Parse a header from raw header-section bytes (everything after the 8-byte length). -/
def parseHeaderBytes (sourceFile : String) (dataOffset : Nat) (headerBytes : ByteArray) :
    Except String (Array TensorSchema) := do
  let headerStr ← match String.fromUTF8? headerBytes with
    | some s => .ok s
    | none => .error "header is not valid UTF-8"
  let header ← match Lean.Json.parse headerStr with
    | .ok j => .ok j
    | .error e => .error s!"header JSON: {e}"
  parseHeaderObject sourceFile dataOffset header

/--
Parse a safetensors header from a whole-file image. `dataOffset` fields point into the same
image, so `tensorBytes` can slice tensor payloads out of it.
-/
def parseHeader (fileBytes : ByteArray) : Except String (Array TensorSchema) := do
  let headerLen64 ← match readUInt64LE fileBytes 0 with
    | some n => .ok n
    | none => .error "missing 8-byte header length"
  let headerLen := headerLen64.toNat
  if 8 + headerLen > fileBytes.size then
    .error s!"header length {headerLen} exceeds file size {fileBytes.size}"
  parseHeaderBytes "" (8 + headerLen) (fileBytes.extract 8 (8 + headerLen))

/-- Read and parse the header of one safetensors file without reading tensor payloads. -/
def readHeader (path : System.FilePath) (sourceFile : String := "") : IO (Array TensorSchema) := do
  IO.FS.withFile path .read fun h => do
    let lenBytes ← h.read 8
    let headerLen ← match readUInt64LE lenBytes 0 with
      | some n => pure n.toNat
      | none => throw <| IO.userError s!"invalid safetensors file '{path}': missing 8-byte header length"
    let headerBytes ← h.read headerLen.toUSize
    if headerBytes.size != headerLen then
      throw <| IO.userError
        s!"invalid safetensors file '{path}': header length {headerLen} exceeds file size"
    match parseHeaderBytes sourceFile (8 + headerLen) headerBytes with
    | .ok tensors => return tensors
    | .error e => throw <| IO.userError s!"invalid safetensors file '{path}': {e}"

/-- Parse `model.safetensors.index.json` into (tensor name, shard file) pairs, in key order. -/
def parseIndexWeightMap (path : System.FilePath) : IO (Array (String × String)) := do
  let text ← IO.FS.readFile path
  let root ← match Lean.Json.parse text with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"invalid index file '{path}': {e}"
  let parse : Except String (Array (String × String)) := do
    let weightMap ← root.getObjVal? "weight_map"
    let kvs ← weightMap.getObj?
    let mut out := #[]
    for (name, shardJson) in kvs.toList do
      let shard ← shardJson.getStr?
      if shard.isEmpty then
        throw s!"tensor '{name}' maps to an empty shard filename"
      out := out.push (name, shard)
    return out
  match parse with
  | .ok pairs => return pairs
  | .error e => throw <| IO.userError s!"invalid index file '{path}': {e}"

/-- The tensor schema set of a checkpoint source: a single file or a shard directory. -/
structure Schema where
  source : System.FilePath
  isDirectory : Bool
  tensors : Array TensorSchema
  deriving Repr, Inhabited

namespace Schema

/-- Look up a tensor by exact HF name. -/
def find? (schema : Schema) (name : String) : Option TensorSchema :=
  schema.tensors.findSome? fun t => if t.name == name then some t else none

/-- The file holding tensor `t`: the source itself, or `source / sourceFile` for shards. -/
def tensorPath (schema : Schema) (t : TensorSchema) : System.FilePath :=
  if schema.isDirectory then schema.source / t.sourceFile else schema.source

end Schema

private def ensureUniqueNames (tensors : Array TensorSchema) (context : String) : IO Unit := do
  let mut seen : Array String := #[]
  for t in tensors do
    if seen.contains t.name then
      throw <| IO.userError s!"duplicate tensor name '{t.name}' in {context}"
    seen := seen.push t.name

/-- Introspect a single-file checkpoint or a sharded checkpoint directory. Directory sources use
`model.safetensors.index.json` when present (and validate its shard references); otherwise every
`*.safetensors` file in the directory is read. -/
def introspect (source : System.FilePath) : IO Schema := do
  if !(← source.pathExists) then
    throw <| IO.userError s!"safetensors source does not exist: {source}"
  if ← source.isDir then
    let indexPath := source / "model.safetensors.index.json"
    let tensors ←
      if ← indexPath.pathExists then
        let weightMap ← parseIndexWeightMap indexPath
        let mut shardHeaders : Array (String × Array TensorSchema) := #[]
        for (_, shard) in weightMap do
          unless shardHeaders.any (·.1 == shard) do
            let shardPath := source / shard
            if !(← shardPath.pathExists) then
              throw <| IO.userError
                s!"invalid index file '{indexPath}': referenced shard does not exist: '{shard}'"
            shardHeaders := shardHeaders.push (shard, ← readHeader shardPath shard)
        weightMap.mapM fun (name, shard) =>
          match shardHeaders.findSome? (fun (s, ts) => ts.findSome? fun t =>
            if s == shard && t.name == name then some t else none) with
          | some t => pure t
          | none => throw <| IO.userError
              s!"invalid index file '{indexPath}': tensor '{name}' not found in shard '{shard}'"
      else
        let entries ← source.readDir
        let shardFiles := entries.map (·.fileName) |>.filter (·.endsWith ".safetensors") |>.qsort (· < ·)
        if shardFiles.isEmpty then
          throw <| IO.userError s!"no '.safetensors' files found in directory '{source}'"
        let mut tensors := #[]
        for shard in shardFiles do
          tensors := tensors ++ (← readHeader (source / shard) shard)
        pure tensors
    ensureUniqueNames tensors s!"source '{source}'"
    return { source, isDirectory := true, tensors }
  else
    let tensors ← readHeader source
    ensureUniqueNames tensors s!"file '{source}'"
    return { source, isDirectory := false, tensors }

/-! ## Payload-preserving shard manifests -/

/-- One validated safetensors data section. `filename` is empty for a single-file schema;
`dataOffset` bytes of header precede the payload and tensor offsets are relative to it. -/
structure PayloadShard where
  filename : String
  dataOffset : Nat
  byteSize : Nat
  tensors : Array TensorSchema
  deriving Repr, Inhabited

/-- A tensor address suitable for an address-free persistent-kernel descriptor. -/
structure PayloadTensorRef where
  shard : UInt32
  byteOffset : UInt64
  tensor : TensorSchema
  deriving Repr, Inhabited

/-- Stable, lexically indexed shard payloads for one introspected source. Uploading these payloads
without repacking makes every `PayloadTensorRef.byteOffset` valid on device. -/
structure PayloadManifest where
  schema : Schema
  shards : Array PayloadShard
  deriving Repr, Inhabited

/--
Selected payload tensors repacked into CUDA buffers. Every retained tensor begins at the requested
byte alignment, and `PayloadTensorRef.shard` indexes `buffers` rather than the source files. This
lets model loaders preserve exact tensor bytes without inheriting an external file format's tensor
placement or retaining unused payloads.
-/
structure PackedPayload where
  buffers : Array (Cuda.Buffer UInt8)
  tensors : Array PayloadTensorRef
  alignment : Nat
  byteSize : Nat

namespace PackedPayload

/-- Find a selected tensor at its repacked device address. -/
def findTensor? (payload : PackedPayload) (name : String) : Option PayloadTensorRef :=
  payload.tensors.findSome? fun tensor =>
    if tensor.tensor.name == name then some tensor else none

end PackedPayload

private def payloadFilenames (schema : Schema) : Array String := Id.run do
  if !schema.isDirectory then
    return #[""]
  let mut names := #[]
  for tensor in schema.tensors do
    if !tensor.sourceFile.isEmpty && !names.contains tensor.sourceFile then
      names := names.push tensor.sourceFile
  return names.qsort (· < ·)

private def payloadPath (schema : Schema) (filename : String) : System.FilePath :=
  if schema.isDirectory then schema.source / filename else schema.source

/-- Validate each source file as one contiguous data payload and assign stable shard indices.
This is the model-scale counterpart of `loadTensor`: tensor bytes are not copied or repacked. -/
def Schema.payloadManifest (schema : Schema) : IO PayloadManifest := do
  if schema.tensors.isEmpty then
    throw <| IO.userError s!"safetensors source '{schema.source}' contains no tensors"
  let filenames := payloadFilenames schema
  let mut shards := #[]
  for filename in filenames do
    let tensors := schema.tensors.filter fun tensor =>
      if schema.isDirectory then tensor.sourceFile == filename else true
    let some first := tensors[0]?
      | throw <| IO.userError s!"safetensors payload '{filename}' contains no tensors"
    unless tensors.all (·.dataOffset == first.dataOffset) do
      throw <| IO.userError s!"safetensors payload '{filename}' has inconsistent data offsets"
    let sorted := tensors.qsort fun left right => left.dataBegin < right.dataBegin
    let mut cursor := 0
    for tensor in sorted do
      unless tensor.dataBegin == cursor do
        throw <| IO.userError
          s!"safetensors payload '{filename}' is not contiguous at tensor '{tensor.name}': expected offset {cursor}, found {tensor.dataBegin}"
      cursor := tensor.dataEnd
    let path := payloadPath schema filename
    let metadata ← path.metadata
    unless metadata.byteSize.toNat == first.dataOffset + cursor do
      throw <| IO.userError
        s!"safetensors payload '{path}' has size {metadata.byteSize.toNat}, expected {first.dataOffset + cursor}"
    shards := shards.push {
      filename, dataOffset := first.dataOffset, byteSize := cursor, tensors := sorted
    }
  return { schema, shards }

/-- Resolve a tensor to its stable payload shard index and relative byte offset. -/
def PayloadManifest.findTensor? (manifest : PayloadManifest) (name : String) :
    Option PayloadTensorRef := Id.run do
  for shardIndex in [:manifest.shards.size] do
    let shard := manifest.shards[shardIndex]!
    if let some tensor := shard.tensors.find? (·.name == name) then
      return some {
        shard := shardIndex.toUInt32
        byteOffset := tensor.dataBegin.toUInt64
        tensor
      }
  return none

private partial def readPayloadExact (handle : IO.FS.Handle) (remaining : Nat)
    (buffer : ByteArray) : IO ByteArray := do
  if remaining == 0 then
    return buffer
  let request := min remaining (64 * 1024 * 1024)
  let chunk ← handle.read request.toUSize
  if chunk.isEmpty then
    throw <| IO.userError s!"unexpected end of safetensors payload with {remaining} bytes left"
  readPayloadExact handle (remaining - chunk.size) (buffer ++ chunk)

private partial def discardPayloadExact (handle : IO.FS.Handle) (remaining : Nat) : IO Unit := do
  if remaining != 0 then
    let request := min remaining (64 * 1024 * 1024)
    let chunk ← handle.read request.toUSize
    if chunk.isEmpty then
      throw <| IO.userError
        s!"unexpected end of safetensors file while skipping {remaining} header bytes"
    discardPayloadExact handle (remaining - chunk.size)

/-- Maximum temporary host allocation used while uploading a model-scale payload. -/
def defaultPayloadUploadChunkBytes : Nat := 64 * 1024 * 1024

private partial def uploadPayloadExact (handle : IO.FS.Handle) (buffer : Cuda.Buffer UInt8)
    (stream : Cuda.Stream) (remaining destinationOffset chunkBytes : Nat) : IO Unit := do
  if remaining == 0 then
    stream.synchronize
    return
  let request := min remaining chunkBytes
  let chunk ← handle.read request.toUSize
  if chunk.isEmpty then
    throw <| IO.userError s!"unexpected end of safetensors payload with {remaining} bytes left"
  Cuda.Buffer.copyFromAt buffer chunk destinationOffset.toUSize stream
  uploadPayloadExact handle buffer stream (remaining - chunk.size)
    (destinationOffset + chunk.size) chunkBytes

private partial def uploadPayloadRegionExact (handle : IO.FS.Handle)
    (buffer : Cuda.Buffer UInt8) (stream : Cuda.Stream)
    (remaining destinationOffset chunkBytes : Nat) : IO Unit := do
  if remaining == 0 then
    return
  let request := min remaining chunkBytes
  let chunk ← handle.read request.toUSize
  if chunk.isEmpty then
    throw <| IO.userError s!"unexpected end of selected safetensors payload with {remaining} bytes left"
  Cuda.Buffer.copyFromAt buffer chunk destinationOffset.toUSize stream
  uploadPayloadRegionExact handle buffer stream (remaining - chunk.size)
    (destinationOffset + chunk.size) chunkBytes

private def alignUp (offset alignment : Nat) : Nat :=
  ((offset + alignment - 1) / alignment) * alignment

/--
Stream a selected tensor set into bounded temporary host storage and CUDA buffers, preserving exact
bytes while aligning every destination tensor independently. Source shards are consumed once in
payload order; unselected bytes are discarded instead of allocated on the device. Empty source
shards are omitted and packed shard indices are dense.
-/
def PayloadManifest.loadPacked (manifest : PayloadManifest) (names : Array String)
    (stream : Cuda.Stream) (alignment : Nat := 128)
    (chunkBytes : Nat := defaultPayloadUploadChunkBytes) : IO PackedPayload := do
  if names.isEmpty then
    throw <| IO.userError "packed safetensors payload requires at least one tensor"
  if alignment == 0 then
    throw <| IO.userError "packed safetensors payload alignment must be positive"
  if chunkBytes == 0 then
    throw <| IO.userError "packed safetensors payload upload chunk size must be positive"
  let mut uniqueNames : Array String := #[]
  for name in names do
    if uniqueNames.contains name then
      throw <| IO.userError s!"packed safetensors payload contains duplicate tensor '{name}'"
    unless (manifest.findTensor? name).isSome do
      throw <| IO.userError s!"packed safetensors payload tensor is missing: '{name}'"
    uniqueNames := uniqueNames.push name
  let mut buffers : Array (Cuda.Buffer UInt8) := #[]
  let mut packedTensors : Array PayloadTensorRef := #[]
  let mut totalBytes := 0
  for sourceIndex in [:manifest.shards.size] do
    let shard := manifest.shards[sourceIndex]!
    let selected := shard.tensors.filter fun tensor => uniqueNames.contains tensor.name
    unless selected.isEmpty do
      let packedShardIndex := buffers.size.toUInt32
      let mut shardBytes := 0
      let mut shardTensors : Array PayloadTensorRef := #[]
      for tensor in selected do
        let byteOffset := alignUp shardBytes alignment
        shardTensors := shardTensors.push {
          shard := packedShardIndex
          byteOffset := byteOffset.toUInt64
          tensor
        }
        shardBytes := byteOffset + tensor.byteSize
      let buffer ← Cuda.Buffer.alloc UInt8 shardBytes.toUSize
      let path := payloadPath manifest.schema shard.filename
      IO.FS.withFile path .read fun handle => do
        discardPayloadExact handle shard.dataOffset
        for tensor in shard.tensors do
          match shardTensors.findSome? fun packed =>
              if packed.tensor.name == tensor.name then some packed else none with
          | some packed =>
            uploadPayloadRegionExact handle buffer stream tensor.byteSize
              packed.byteOffset.toNat chunkBytes
          | none =>
            discardPayloadExact handle tensor.byteSize
      buffers := buffers.push buffer
      packedTensors := packedTensors ++ shardTensors
      totalBytes := totalBytes + shardBytes
  stream.synchronize
  unless packedTensors.size == uniqueNames.size do
    throw <| IO.userError
      s!"packed safetensors payload retained {packedTensors.size} tensors, expected {uniqueNames.size}"
  return { buffers, tensors := packedTensors, alignment, byteSize := totalBytes }

/-- Read one whole validated data payload, excluding its safetensors header. -/
def PayloadManifest.readShard (manifest : PayloadManifest) (index : Nat) : IO ByteArray := do
  let some shard := manifest.shards[index]?
    | throw <| IO.userError s!"safetensors payload shard index {index} is out of range"
  let path := payloadPath manifest.schema shard.filename
  let handle ← IO.FS.Handle.mk path .read
  discardPayloadExact handle shard.dataOffset
  readPayloadExact handle shard.byteSize .empty

/-- Upload one validated payload without materializing it as a model-scale `ByteArray`. At most
`chunkBytes` pageable host bytes are live at once, while the returned device buffer preserves the
manifest's relative tensor offsets exactly. -/
def PayloadManifest.loadShard (manifest : PayloadManifest) (index : Nat)
    (stream : Cuda.Stream) (chunkBytes : Nat := defaultPayloadUploadChunkBytes) :
    IO (PayloadShard × Cuda.Buffer UInt8) := do
  if chunkBytes == 0 then
    throw <| IO.userError "safetensors payload upload chunk size must be positive"
  let some shard := manifest.shards[index]?
    | throw <| IO.userError s!"safetensors payload shard index {index} is out of range"
  let path := payloadPath manifest.schema shard.filename
  let buffer ← Cuda.Buffer.alloc UInt8 shard.byteSize.toUSize
  IO.FS.withFile path .read fun handle => do
    discardPayloadExact handle shard.dataOffset
    uploadPayloadExact handle buffer stream shard.byteSize 0 chunkBytes
  return (shard, buffer)

/-- Extract one tensor's raw bytes from an already-read shard image. -/
def tensorBytes (fileBytes : ByteArray) (t : TensorSchema) : Except String ByteArray :=
  if t.dataEnd < t.dataBegin then
    .error s!"tensor '{t.name}': data_offsets [{t.dataBegin}, {t.dataEnd}) are reversed"
  else
    let start := t.dataOffset + t.dataBegin
    let stop := t.dataOffset + t.dataEnd
    if stop > fileBytes.size then
      .error s!"tensor '{t.name}': byte range [{start}, {stop}) exceeds file size {fileBytes.size}"
    else
      .ok (fileBytes.extract start stop)

private def ofExceptIO (context : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError s!"{context}: {e}"

/-- Read one tensor's raw bytes from its shard file. -/
def readTensorBytes (path : System.FilePath) (t : TensorSchema) : IO ByteArray := do
  let bytes ← IO.FS.readBinFile path
  ofExceptIO s!"'{path}'" (tensorBytes bytes t)

/-- Upload raw bytes to a fresh device buffer, waiting for the copy to land. -/
def uploadBytes (bytes : ByteArray) (stream : Cuda.Stream) : IO (Cuda.Buffer UInt8) := do
  let buf ← Cuda.Buffer.alloc UInt8 bytes.size.toUSize
  Cuda.Buffer.copyFrom buf bytes stream
  stream.synchronize
  return buf

/-- Upload every validated payload in stable shard order using bounded host staging. The returned
shard metadata and device buffer pairs preserve `PayloadTensorRef` addressing exactly. -/
def PayloadManifest.loadShards (manifest : PayloadManifest) (stream : Cuda.Stream)
    (chunkBytes : Nat := defaultPayloadUploadChunkBytes) :
    IO (Array (PayloadShard × Cuda.Buffer UInt8)) := do
  let mut loaded := #[]
  for index in [:manifest.shards.size] do
    loaded := loaded.push (← manifest.loadShard index stream chunkBytes)
  return loaded

/-- Load one tensor into a fresh device buffer: raw file bytes, no dtype conversion. -/
def loadTensor (schema : Schema) (name : String) (stream : Cuda.Stream) :
    IO (TensorSchema × Cuda.Buffer UInt8) := do
  let t ← match schema.find? name with
    | some t => pure t
    | none => throw <| IO.userError s!"tensor '{name}' not found in '{schema.source}'"
  let bytes ← readTensorBytes (schema.tensorPath t) t
  return (t, ← uploadBytes bytes stream)

/-- Load every tensor of a schema, reading each shard file from disk only once. -/
def loadTensors (schema : Schema) (stream : Cuda.Stream) :
    IO (Array (TensorSchema × Cuda.Buffer UInt8)) := do
  let mut files : Array (System.FilePath × ByteArray) := #[]
  let mut out := #[]
  for t in schema.tensors do
    let path := schema.tensorPath t
    let bytes ← match files.findSome? (fun (p, b) => if p == path then some b else none) with
      | some b => pure b
      | none =>
        let b ← IO.FS.readBinFile path
        files := files.push (path, b)
        pure b
    out := out.push (t, ← uploadBytes (← ofExceptIO s!"'{path}'" (tensorBytes bytes t)) stream)
  return out

/-! ## Qwen3.8 HF-name registry -/

/-- Typed weight families of the Qwen3.8 text model. -/
inductive WeightFamily where
  | embed | finalNorm | lmHead
  | inputNorm (layer : Nat) | postAttnNorm (layer : Nat)
  | deltanetQKV (layer : Nat) | deltanetZ (layer : Nat) | deltanetB (layer : Nat)
  | deltanetA (layer : Nat) | deltanetConv (layer : Nat) | deltanetALog (layer : Nat)
  | deltanetDtBias (layer : Nat) | deltanetNorm (layer : Nat) | deltanetOut (layer : Nat)
  | attnQ (layer : Nat) | attnK (layer : Nat) | attnV (layer : Nat) | attnO (layer : Nat)
  | attnQNorm (layer : Nat) | attnKNorm (layer : Nat)
  | mlpGate (layer : Nat) | mlpUp (layer : Nat) | mlpDown (layer : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

/-- The HuggingFace tensor name of a weight family, matching the published
`Qwen3.8-27B/model.safetensors.index.json`. Note that `A_log` and `dt_bias` carry no `.weight`. -/
def WeightFamily.hfName : WeightFamily → String
  | .embed => "model.language_model.embed_tokens.weight"
  | .finalNorm => "model.language_model.norm.weight"
  | .lmHead => "lm_head.weight"
  | .inputNorm l => s!"model.language_model.layers.{l}.input_layernorm.weight"
  | .postAttnNorm l => s!"model.language_model.layers.{l}.post_attention_layernorm.weight"
  | .deltanetQKV l => s!"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight"
  | .deltanetZ l => s!"model.language_model.layers.{l}.linear_attn.in_proj_z.weight"
  | .deltanetB l => s!"model.language_model.layers.{l}.linear_attn.in_proj_b.weight"
  | .deltanetA l => s!"model.language_model.layers.{l}.linear_attn.in_proj_a.weight"
  | .deltanetConv l => s!"model.language_model.layers.{l}.linear_attn.conv1d.weight"
  | .deltanetALog l => s!"model.language_model.layers.{l}.linear_attn.A_log"
  | .deltanetDtBias l => s!"model.language_model.layers.{l}.linear_attn.dt_bias"
  | .deltanetNorm l => s!"model.language_model.layers.{l}.linear_attn.norm.weight"
  | .deltanetOut l => s!"model.language_model.layers.{l}.linear_attn.out_proj.weight"
  | .attnQ l => s!"model.language_model.layers.{l}.self_attn.q_proj.weight"
  | .attnK l => s!"model.language_model.layers.{l}.self_attn.k_proj.weight"
  | .attnV l => s!"model.language_model.layers.{l}.self_attn.v_proj.weight"
  | .attnO l => s!"model.language_model.layers.{l}.self_attn.o_proj.weight"
  | .attnQNorm l => s!"model.language_model.layers.{l}.self_attn.q_norm.weight"
  | .attnKNorm l => s!"model.language_model.layers.{l}.self_attn.k_norm.weight"
  | .mlpGate l => s!"model.language_model.layers.{l}.mlp.gate_proj.weight"
  | .mlpUp l => s!"model.language_model.layers.{l}.mlp.up_proj.weight"
  | .mlpDown l => s!"model.language_model.layers.{l}.mlp.down_proj.weight"

/-- Parse an HF tensor name back into a weight family; `none` when the name follows no Qwen3.8
text-model pattern (including out-of-scope `model.visual.*` / `mtp.*` names). -/
def WeightFamily.ofHFName? (name : String) : Option WeightFamily :=
  match name with
  | "model.language_model.embed_tokens.weight" => some .embed
  | "model.language_model.norm.weight" => some .finalNorm
  | "lm_head.weight" => some .lmHead
  | _ =>
    match name.splitOn "." with
    | "model" :: "language_model" :: "layers" :: layerStr :: suffix =>
      match layerStr.toNat? with
      | none => none
      | some l =>
        match suffix with
        | ["input_layernorm", "weight"] => some (.inputNorm l)
        | ["post_attention_layernorm", "weight"] => some (.postAttnNorm l)
        | ["linear_attn", "in_proj_qkv", "weight"] => some (.deltanetQKV l)
        | ["linear_attn", "in_proj_z", "weight"] => some (.deltanetZ l)
        | ["linear_attn", "in_proj_b", "weight"] => some (.deltanetB l)
        | ["linear_attn", "in_proj_a", "weight"] => some (.deltanetA l)
        | ["linear_attn", "conv1d", "weight"] => some (.deltanetConv l)
        | ["linear_attn", "A_log"] => some (.deltanetALog l)
        | ["linear_attn", "dt_bias"] => some (.deltanetDtBias l)
        | ["linear_attn", "norm", "weight"] => some (.deltanetNorm l)
        | ["linear_attn", "out_proj", "weight"] => some (.deltanetOut l)
        | ["self_attn", "q_proj", "weight"] => some (.attnQ l)
        | ["self_attn", "k_proj", "weight"] => some (.attnK l)
        | ["self_attn", "v_proj", "weight"] => some (.attnV l)
        | ["self_attn", "o_proj", "weight"] => some (.attnO l)
        | ["self_attn", "q_norm", "weight"] => some (.attnQNorm l)
        | ["self_attn", "k_norm", "weight"] => some (.attnKNorm l)
        | ["mlp", "gate_proj", "weight"] => some (.mlpGate l)
        | ["mlp", "up_proj", "weight"] => some (.mlpUp l)
        | ["mlp", "down_proj", "weight"] => some (.mlpDown l)
        | _ => none
    | _ => none

/-- Layer layout of a Qwen3.8 text model. Layer `i` is full attention iff
`i % fullAttentionPeriod == fullAttentionPeriod - 1` (the 27B and tiny configs use period 4). -/
structure RegistryConfig where
  numLayers : Nat
  fullAttentionPeriod : Nat := 4
  deriving Repr, Inhabited

namespace RegistryConfig

/-- Whether layer `i` is a full-attention layer. -/
def isFullAttention (cfg : RegistryConfig) (i : Nat) : Bool :=
  cfg.fullAttentionPeriod > 0 && i % cfg.fullAttentionPeriod == cfg.fullAttentionPeriod - 1

/-- Every weight family a checkpoint for this configuration must contain, in layer-stack order. -/
def expected (cfg : RegistryConfig) : Array WeightFamily := Id.run do
  let mut out := #[.embed]
  for layer in Array.range cfg.numLayers do
    out := out.push (.inputNorm layer)
    if cfg.isFullAttention layer then
      out := out ++ #[.attnQ layer, .attnK layer, .attnV layer, .attnO layer,
        .attnQNorm layer, .attnKNorm layer]
    else
      out := out ++ #[.deltanetQKV layer, .deltanetZ layer, .deltanetB layer, .deltanetA layer,
        .deltanetConv layer, .deltanetALog layer, .deltanetDtBias layer, .deltanetNorm layer,
        .deltanetOut layer]
    out := out ++ #[.postAttnNorm layer, .mlpGate layer, .mlpUp layer, .mlpDown layer]
  return out ++ #[.finalNorm, .lmHead]

/-- HF names of `expected`. -/
def expectedNames (cfg : RegistryConfig) : Array String := cfg.expected.map (·.hfName)

end RegistryConfig

/-- Outcome of checking a checkpoint schema against the registry. -/
structure CheckReport where
  /-- Expected tensors absent from the schema. -/
  missing : Array String := #[]
  /-- Schema tensors that follow no expected Qwen3.8 pattern for this configuration
  (wrong layer type or unknown name). -/
  unexpected : Array String := #[]
  /-- Schema tensors outside text scope (`model.visual.*`, `mtp.*`); never an error. -/
  outOfScope : Array String := #[]
  deriving Repr, Inhabited

/-- A schema is complete when nothing expected is missing and nothing unknown is present. -/
def CheckReport.isClean (r : CheckReport) : Bool := r.missing.isEmpty && r.unexpected.isEmpty

/-- Check a checkpoint schema against the weight families a configuration expects. -/
def checkComplete (cfg : RegistryConfig) (schema : Schema) : CheckReport := Id.run do
  let expectedNames := cfg.expectedNames
  let mut missing := #[]
  for name in expectedNames do
    unless schema.tensors.any (·.name == name) do
      missing := missing.push name
  let mut unexpected := #[]
  let mut outOfScope := #[]
  for t in schema.tensors do
    if t.name.startsWith "model.visual." || t.name.startsWith "mtp." then
      outOfScope := outOfScope.push t.name
    else if !expectedNames.contains t.name then
      unexpected := unexpected.push t.name
  return { missing, unexpected, outOfScope }

end Cuda.Qwen36.SafeTensors
