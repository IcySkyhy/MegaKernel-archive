/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Qwen36Checkpoint

namespace Qwen36PayloadTests

open Qwen36.Checkpoint

private def pushUInt64LE (bytes : ByteArray) (value : UInt64) : ByteArray := Id.run do
  let mut result := bytes
  for shift in [0:8] do
    result := result.push (value >>> (shift * 8).toUInt64).toUInt8
  return result

@[test]
def shardPayloadRoundTrip : IO Unit := do
  let text :=
    "{\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}," ++
    "\"bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[12,20]}}"
  let payload := ByteArray.mk #[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  IO.FS.withTempFile fun handle path => do
    let file := pushUInt64LE .empty text.toUTF8.size.toUInt64 ++ text.toUTF8 ++ payload
    handle.write file
    handle.flush
    let header ← readHeader path
    LeanTest.assertEqual header.dataBytes payload.size
    LeanTest.assertEqual (← readShardPayload path header) payload

@[test]
def shardPayloadChunkStream : IO Unit := do
  let text :=
    "{\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}," ++
    "\"bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[12,20]}}"
  let payload := ByteArray.mk #[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  IO.FS.withTempFile fun handle path => do
    let file := pushUInt64LE .empty text.toUTF8.size.toUInt64 ++ text.toUTF8 ++ payload
    handle.write file
    handle.flush
    let header ← readHeader path
    let offsets ← IO.mkRef (#[] : Array Nat)
    let streamed ← IO.mkRef ByteArray.empty
    forEachShardPayloadChunk path header (chunkBytes := 7) fun offset chunk => do
      offsets.modify (·.push offset)
      streamed.modify (· ++ chunk)
    LeanTest.assertEqual (← offsets.get) #[0, 7, 14]
    LeanTest.assertEqual (← streamed.get) payload

end Qwen36PayloadTests
