/**
 * @file Megakernel.cpp
 * @brief MLIR C interface wrappers for megakernel custom ops.
 *
 * Ops implemented here (all under dl.ascend.* Python namespace):
 *   - global_block_idx_mix -> _mlir_ciface_dtile_global_block_idx_mix
 *   - wait_deps       -> _mlir_ciface_dtile_wait_deps
 *   - notify_tile_done -> _mlir_ciface_dtile_notify_tile_done
 *   - query_ts_condition -> _mlir_ciface_dtile_query_ts_condition
 *   - set_cond       -> _mlir_ciface_dtile_set_cond
 *
 * NOTE: wait_deps / notify_tile_done are intentionally left as no-ops for now.
 * Cache coherence (dcci) and spin-wait logic will be implemented later; for
 * the current integration phase the caller side handles dcci explicitly.
 * query_ts_condition and set_cond are implemented since they are simple
 * register/intrinsic accesses with no scoreboard logic.
 */

#include "Utils.h"  // memref_t<>

// --- Internal helpers ---

static __aicore__ __attribute__((always_inline)) int32_t
dtile_global_block_idx_mix() {
// assume mix is mandatory
#ifdef __DAV_VEC__
    return get_block_idx() * get_subblockdim() + get_subblockid() +
           get_block_num();
#else
    return get_block_idx();
#endif
}

static __aicore__ __attribute__((always_inline)) uint64_t
dtile_query_ts_condition() {
    uint64_t val = 0;
    __asm__ volatile("MOV %0, DATA_MAIN_BASE\n" : "+l"(val));
    return val;
}

static __aicore__ __attribute__((always_inline)) void
dtile_set_cond(uint64_t cond) {
    set_cond(cond);
}

static __aicore__ __attribute__((always_inline)) void
dtile_set_status(__gm__ uint32_t *ptr, uint32_t val) {
#ifdef __DAV_VEC__
    *ptr = val | (1 << 16);
#else
    *ptr = val | (2 << 16);
#endif
    dcci(ptr, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

static __aicore__ __attribute__((always_inline)) void
dtile_st_io_u32(__gm__ uint32_t *ptr, uint32_t val) {
    *ptr = val;
}

static __aicore__ __attribute__((always_inline)) void
dtile_st_io_u16(__gm__ uint16_t *ptr, uint16_t val) {
    *ptr = val;
}

static __aicore__ __attribute__((always_inline)) void
dtile_st_io_u64(__gm__ uint64_t *ptr, uint64_t val) {
    *ptr = val;
}

static __aicore__ __attribute__((always_inline)) uint32_t
dtile_ld_io_u32(__gm__ uint32_t *ptr) {
    return *ptr;
}

// --- MLIR C interface wrappers ---

#ifdef __cplusplus
extern "C" {
#endif

__aicore__ __attribute__((always_inline)) int32_t
_mlir_ciface_dtile_global_block_idx_mix() {
    return dtile_global_block_idx_mix();
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_wait_deps(memref_t<__gm__ uint8_t, 1> dependInfoIn) {
    // No-op for now: scoreboard spin-wait + dcci will be implemented later.
    (void)dependInfoIn;
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_notify_tile_done(memref_t<__gm__ uint8_t, 1> score_board,
                                   int32_t tile_index) {
    // No-op for now: scoreboard flag write + dcci will be implemented later.
    (void)score_board;
    (void)tile_index;
}

__aicore__ __attribute__((always_inline)) uint64_t
_mlir_ciface_dtile_query_ts_condition() {
    return dtile_query_ts_condition();
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_set_cond(uint64_t cond) {
    dtile_set_cond(cond);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_set_status(uint64_t addr, int32_t val) {
    auto gm_ptr = reinterpret_cast<__gm__ uint32_t *>(addr);
    dtile_set_status(gm_ptr, static_cast<uint32_t>(val));
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_st_io_u32(uint64_t addr, int32_t val) {
    auto gm_ptr = reinterpret_cast<__gm__ uint32_t *>(addr);
    dtile_st_io_u32(gm_ptr, static_cast<uint32_t>(val));
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_st_io_u16(uint64_t addr, int32_t val) {
    auto gm_ptr = reinterpret_cast<__gm__ uint16_t *>(addr);
    dtile_st_io_u16(gm_ptr, static_cast<uint16_t>(val));
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_dtile_st_io_u64(uint64_t addr, int64_t val) {
    auto gm_ptr = reinterpret_cast<__gm__ uint64_t *>(addr);
    dtile_st_io_u64(gm_ptr, static_cast<uint64_t>(val));
}

__aicore__ __attribute__((always_inline)) int32_t
_mlir_ciface_dtile_ld_io(uint64_t addr) {
    auto gm_ptr = reinterpret_cast<__gm__ uint32_t *>(addr);
    return static_cast<int32_t>(dtile_ld_io_u32(gm_ptr));
}

#ifdef __cplusplus
}
#endif
