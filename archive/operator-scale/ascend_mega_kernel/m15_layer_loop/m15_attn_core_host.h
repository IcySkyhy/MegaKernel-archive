// ============================================================
// m15_attn_core_host.h —— M101 host 侧：attention 核心段的**独立设备验证路**（数据生成 + 落盘 + 负向对照）
//
// 只被 `m15_attn_core.asc` include（在 `kernel_operator.h` / `acl/acl.h` / `<cstdio>` 等之后）。
// 本文件**不自带 include**，前提同 `m15_attn_prolog_host.h`。
//
// 为什么 host 侧要**逐字复刻 donor 的 `main()`**（`m10_attn_decode.asc:1120-1339`）：
//   本 mission 的第三步要求「抽取后的读数必须与 M10 已归档读数在 256/300 档一致」。最直接的
//   判据不是"看起来差不多"，而是**同一组输入、同一套 dump 文件名、同一份判据脚本**：
//     · 输入生成 `GenQKV` / `GenMaskCol`（Hash3 + FloatToBf16，salt = 0xA100 + seq）逐字复刻 ⇒
//       q/k/v **字节**与 donor 相同（`m10_attn_decode/data/m10_case_s*_{q,k,v}.bin` 可直接对拍）；
//     · dump 文件名沿用 donor 的 `m10_case_s<seq>_*.bin` ⇒ donor 的 `m10_attn_decode/check_ref.py`
//       **原样**就能跑（它把前缀写死在 `load_case()` 里，见该文件 `pre = f"m10_case_s{seq}_"`），
//       不需要在本 mission 里复制一份判据（判据必须独立于被测实现 —— 复制会带来"两份判据"漂移）；
//     · 设备侧只有 kernel 主体换成了抽取件 `M15AC::AttnCoreBody`（其余 buffer 尺寸/毒值/launch 参数
//       与 donor 逐字相同，见 `m10_attn_decode.asc:1180-1286`）。
//
// 判据（**本文件不重写判据**，只负责跑 + 对拍字节 + 打印读数）：
//   · 判定项 A–F / guard / 退出码三态 ⇒ donor 的 `check_ref.py`（cwd 里跑）；
//   · 抽取等价性 ⇒ `M15AC_REFDIR` 指向 `m10_attn_decode/data` 时逐文件字节比对（本文件实现）。
//
// 负向对照（docs/17 §4 的"被测对象弄坏 ⇒ 判据必变红"）：
//   `negkv` 档 = **实参错位**：把 K 与 V 两个 GM 指针在调用点互换。抽出的段一字未改，
//   变的是调用点 ⇒ 若判据 **A/C/E/F** 不红，说明判据对"K/V 通路"没有咬合力。
//   **D 不在此列**（复审 r1-F5 更正）：判据 D 咬的是"未用槽位恒 0 + 尾 tile 无效列恒 0"这条
//   **结构性**性质，K/V 换位置改变不了它 ⇒ 本 mission 的 negkv 档里它仍 PASS
//   （README §3.3 记的也正是 A/C/E/F 四项 —— 这是实测，不是"应当"）。
//   选它的理由：它不依赖改 header（header 是生成物 `--check` 会抓），也不需要重编译。
// ============================================================
#ifndef M15_ATTN_CORE_HOST_H
#define M15_ATTN_CORE_HOST_H

namespace M15ACHost {

// ---- donor host 侧常量（`m10_attn_decode.asc:1141-1211` 的尺寸，逐字）----
constexpr uint32_t H_N2 = 2;
constexpr uint32_t H_G = 12;
constexpr uint32_t H_M_PAD = 16;
constexpr uint32_t H_HD = 256;
constexpr uint32_t H_S2T = 256;
constexpr uint32_t H_SPLITS = 14;
constexpr uint32_t H_MAX_SEQ = 4096;
constexpr uint32_t H_SCRATCH_SETUP = 28;   // donor 要求 AIC 数 == 28（14 split × 2 KV 头）

// ---- donor host 侧的确定性生成（逐字复刻；改一处必须改两处）----
static uint32_t Hash3(uint32_t i, uint32_t j, uint32_t salt)
{
    uint32_t h = i * 2654435761u + j * 40503u + salt * 97u + 0x9E3779B9u;
    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

static uint16_t FloatToBf16(float f)
{
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    const uint32_t roundingBias = ((x >> 16) & 1) + 0x7FFF;
    return (uint16_t)((x + roundingBias) >> 16);
}

static float Bf16ToFloat(uint16_t h)
{
    uint32_t x = ((uint32_t)h) << 16;
    float f;
    __builtin_memcpy(&f, &x, 4);
    return f;
}

static void GenQKV(uint16_t* q, uint16_t* k, uint16_t* v, uint32_t seq, uint32_t seqPad, uint32_t salt,
                   uint32_t synth = 0)
{
    for (uint32_t n2 = 0; n2 < H_N2; ++n2) {
        for (uint32_t r = 0; r < H_M_PAD; ++r) {
            for (uint32_t c = 0; c < H_HD; ++c) {
                float val = 0.0f;
                if (r < H_G) {
                    if (synth == 2) {
                        val = (c == r) ? 1.0f : 0.0f;
                    } else if (synth == 3 || synth == 4) {
                        val = (r == 0 && c == 0) ? 1.0f : 0.0f;
                    } else {
                        val = (float)(Hash3(n2 * 16 + r, c, salt) & 0xFFFF) / 65535.0f * 1.6f - 0.8f;
                    }
                }
                q[(n2 * H_M_PAD + r) * H_HD + c] = FloatToBf16(val);
            }
        }
        for (uint32_t r = 0; r < seqPad; ++r) {
            for (uint32_t c = 0; c < H_HD; ++c) {
                float kv = 0.0f;
                float vv = 0.0f;
                if (r < seq) {
                    if (synth == 4) {
                        kv = 1.0f;
                    } else if (synth != 0) {
                        kv = (c == (r % 256)) ? 1.0f : 0.0f;
                    } else {
                        kv = (float)(Hash3(n2 * 4096 + r, c, salt + 11) & 0xFFFF) / 65535.0f * 1.6f - 0.8f;
                    }
                    if (synth == 5) {
                        vv = (c == (r % 256)) ? 1.0f : 0.0f;
                    } else {
                        vv = (float)(Hash3(n2 * 4096 + r, c, salt + 23) & 0xFFFF) / 65535.0f * 1.6f - 0.8f;
                    }
                }
                k[(n2 * seqPad + r) * H_HD + c] = FloatToBf16(kv);
                v[(n2 * seqPad + r) * H_HD + c] = FloatToBf16(vv);
            }
        }
    }
}

static void GenMaskCol(float* mask, uint32_t seq)
{
    const uint32_t validTail = seq - (seq / 256) * 256;
    for (uint32_t c = 0; c < 256; ++c) {
        mask[c] = (validTail == 0 || c < validTail) ? 0.0f : -3.38953139e38f;
    }
}

static bool DumpBin(const char* path, const void* data, size_t bytes)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("[M101][FAIL] cannot write %s\n", path);
        return false;
    }
    const size_t w = fwrite(data, 1, bytes, f);
    fclose(f);
    return w == bytes;
}

// ---- 落盘 + 指纹（M101 r2 / 复审 F1 的可审计性修复）----
// 为什么要指纹：`.bin` 是运行产物、不入库（本节 `.gitignore` 忽略 `*.bin`）。只把**读数**留在
// 入库的日志里、而**输入张量**（attn/gate/W 由公式确定）不留任何可核对的锚点，评审就只能重跑设备。
// 指纹连同长度打进**入库的日志**，于是离线复核脚本可以按公式重生成输入、用指纹证明
// "与我算的是同一批字节"，再独立复算判据。
//
// **算法 = 标准 FNV-1a 64**（复审 r2-N1 更正）：offset basis `14695981039346656037`
// （= 0xCBF29CE484222325）、prime `1099511628211`（= 0x100000001B3）、每字节
// `h = (h ^ byte) * prime`（mod 2^64）。**r1 提交里那个常量少了一位**（`…603` 而非 `…6037`），
// 名字却叫 `Fnv1a64` ⇒ 用标准算法实现的第三方复现不出来。已按复审建议改**常量**而不是改名。
// 对照向量（第三方可用它自查实现）：`FNV1a64(b"") = 0xcbf29ce484222325`、
// `FNV1a64(b"a") = 0xaf63dc4c8601ec8c`。
static uint64_t Fnv1a64(const void* p, size_t n)
{
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 14695981039346656037ull;   // 标准 FNV-1a 64 offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ull;              // 标准 FNV-1a 64 prime
    }
    return h;
}

// 落盘并打印指纹；`label` 用于区分"同一份 buffer 在不同时点落盘"（防止同名覆盖造成误读 —— F1 的成因）
static bool DumpAndPrint(const char* label, const char* path, const void* p, size_t n)
{
    const bool ok = DumpBin(path, p, n);
    printf("[M101][fp] %-20s file=%-28s len=%-9zu fnv1a64=%016llx%s\n", label, path, n,
           (unsigned long long)Fnv1a64(p, n), ok ? "" : "  **写失败**");
    return ok;
}

// ---- 与 donor 归档逐字节对拍（抽取等价性的直接见证）----
// 返回 0 = 逐字节相同；>0 = 首个不同字节的下标 + 1；-1 = 读不到参考文件
static long CmpWithRef(const char* refDir, const char* name, const void* got, size_t bytes)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", refDir, name);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    static uint8_t ref[1 << 21];
    if (bytes > sizeof(ref)) {
        fclose(f);
        printf("[M101][FAIL] %s 参考缓冲太小（%zu）\n", name, bytes);
        return -2;
    }
    const size_t n = fread(ref, 1, sizeof(ref), f);
    fclose(f);
    if (n != bytes) {
        printf("[M101][bincmp] %s 尺寸不同：ref %zu vs got %zu\n", name, n, bytes);
        return -3;
    }
    const uint8_t* g = (const uint8_t*)got;
    for (size_t i = 0; i < bytes; ++i) {
        if (ref[i] != g[i]) return (long)i + 1;
    }
    return 0;
}

namespace detail {

struct RunCtx {
    aclrtStream stream = nullptr;
    // device
    void* dQ = nullptr; void* dK = nullptr; void* dV = nullptr; void* dOut = nullptr; void* dSeq = nullptr;
    void* dWsA = nullptr; void* dWsM = nullptr; void* dWsS = nullptr; void* dMask = nullptr;
    void* dDbgS = nullptr; void* dDbgP = nullptr; void* dDbgC = nullptr; void* dDbgC2 = nullptr;
    void* dCfg = nullptr; void* dGP = nullptr;
    // host
    uint16_t* hQ = nullptr; uint16_t* hK = nullptr; uint16_t* hV = nullptr; uint16_t* hOut = nullptr;
    uint32_t* hSeq = nullptr; float* hMask = nullptr;
    float* hWsA = nullptr; float* hWsM = nullptr; float* hWsS = nullptr;
    float* hDbgS = nullptr; uint16_t* hDbgP = nullptr;
    float* hDbgC = nullptr; float* hDbgC2 = nullptr; uint16_t* hGP = nullptr;
    uint32_t hCfg[24] = {0};
};

constexpr size_t SZ_Q = (size_t)2 * 16 * 256 * 2;
constexpr size_t SZ_KV = (size_t)2 * H_MAX_SEQ * 256 * 2;
constexpr size_t SZ_OUT = (size_t)2 * 12 * 256 * 2;
constexpr size_t SZ_WSACC = (size_t)2 * 14 * 16 * 256 * 4;
constexpr size_t SZ_WSMS = (size_t)2 * 14 * 16 * 4;
constexpr size_t SZ_DBGS = (size_t)56 * 8 * 256 * 4;
constexpr size_t SZ_DBGP = (size_t)56 * 8 * 256 * 2;
constexpr size_t SZ_DBGC = (size_t)28 * 16 * 256 * 4;
constexpr size_t SZ_GP = (size_t)28 * 3 * 16 * 256 * 2;

static bool Alloc(RunCtx& c)
{
    aclrtMalloc(&c.dQ, SZ_Q, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dK, SZ_KV, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dV, SZ_KV, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dOut, SZ_OUT, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dSeq, 64, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dWsA, SZ_WSACC, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dWsM, SZ_WSMS, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dWsS, SZ_WSMS, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dMask, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dDbgS, SZ_DBGS, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dDbgP, SZ_DBGP, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dDbgC, SZ_DBGC, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dDbgC2, SZ_DBGC, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dGP, SZ_GP, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&c.dCfg, 96, ACL_MEM_MALLOC_HUGE_FIRST);
    c.hQ = (uint16_t*)malloc(SZ_Q);
    c.hK = (uint16_t*)malloc(SZ_KV);
    c.hV = (uint16_t*)malloc(SZ_KV);
    c.hOut = (uint16_t*)malloc(SZ_OUT);
    c.hSeq = (uint32_t*)malloc(64);
    c.hMask = (float*)malloc(1024);
    c.hWsA = (float*)malloc(SZ_WSACC);
    c.hWsM = (float*)malloc(SZ_WSMS);
    c.hWsS = (float*)malloc(SZ_WSMS);
    c.hDbgS = (float*)malloc(SZ_DBGS);
    c.hDbgP = (uint16_t*)malloc(SZ_DBGP);
    c.hDbgC = (float*)malloc(SZ_DBGC);
    c.hDbgC2 = (float*)malloc(SZ_DBGC);
    c.hGP = (uint16_t*)malloc(SZ_GP);
    return c.dQ && c.dK && c.dV && c.dOut && c.dSeq && c.dWsA && c.dWsM && c.dWsS && c.dMask &&
           c.dDbgS && c.dDbgP && c.dDbgC && c.dDbgC2 && c.dGP && c.dCfg && c.hQ && c.hK && c.hV &&
           c.hOut && c.hSeq && c.hMask && c.hWsA && c.hWsM && c.hWsS && c.hDbgS && c.hDbgP && c.hDbgC &&
           c.hDbgC2 && c.hGP;
}

static void Free(RunCtx& c)
{
    aclrtFree(c.dQ); aclrtFree(c.dK); aclrtFree(c.dV); aclrtFree(c.dOut); aclrtFree(c.dSeq);
    aclrtFree(c.dWsA); aclrtFree(c.dWsM); aclrtFree(c.dWsS); aclrtFree(c.dMask);
    aclrtFree(c.dDbgS); aclrtFree(c.dDbgP); aclrtFree(c.dDbgC); aclrtFree(c.dDbgC2);
    aclrtFree(c.dGP); aclrtFree(c.dCfg);
    free(c.hQ); free(c.hK); free(c.hV); free(c.hOut); free(c.hSeq); free(c.hMask);
    free(c.hWsA); free(c.hWsM); free(c.hWsS); free(c.hDbgS); free(c.hDbgP); free(c.hDbgC);
    free(c.hDbgC2); free(c.hGP);
}

// `M15AC_L0CFG="qM,qK,qS,qD,qT,kM,kK,kS,kD,kT[,vM,vK,vS,vD,vT]"`（donor 的调试覆盖，逐字）
static void ParseL0Cfg(RunCtx& c)
{
    const char* e = getenv("M15AC_L0CFG");
    if (!e) return;
    uint32_t vals[15] = {0};
    sscanf(e, "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u", &vals[0], &vals[1], &vals[2], &vals[3],
           &vals[4], &vals[5], &vals[6], &vals[7], &vals[8], &vals[9], &vals[10], &vals[11], &vals[12],
           &vals[13], &vals[14]);
    c.hCfg[0] = 1;
    for (int i = 0; i < 5; ++i) c.hCfg[1 + i] = vals[i];
    c.hCfg[6] = 1;
    for (int i = 0; i < 5; ++i) c.hCfg[7 + i] = vals[5 + i];
    if (vals[10] != 0) {
        c.hCfg[12] = 1;
        for (int i = 0; i < 5; ++i) c.hCfg[13 + i] = vals[10 + i];
    }
    printf("[M101] M15AC_L0CFG: Q(%u,%u,%u,%u,T%u) K(%u,%u,%u,%u,T%u) V(%u,%u,%u,%u,T%u)\n", c.hCfg[1],
           c.hCfg[2], c.hCfg[3], c.hCfg[4], c.hCfg[5], c.hCfg[7], c.hCfg[8], c.hCfg[9], c.hCfg[10],
           c.hCfg[11], c.hCfg[13], c.hCfg[14], c.hCfg[15], c.hCfg[16], c.hCfg[17]);
}

// 一次 case：生成 → H2D → launch → 同步 → D2H → 落盘（+可选与 donor 归档对拍）
static int RunCoreCase(RunCtx& c, uint32_t numBlocks, uint32_t seq, uint32_t synth, const char* refDir,
                       bool swapKv, bool dump)
{
    const uint32_t seqPad = (seq + 255) / 256 * 256;
    const uint32_t salt = 0xA100u + seq;
    GenQKV(c.hQ, c.hK, c.hV, seq, seqPad, salt, synth);
    GenMaskCol(c.hMask, seq);
    c.hSeq[0] = seq;

    aclrtMemcpy(c.dQ, SZ_Q, c.hQ, SZ_Q, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(c.dK, SZ_KV, c.hK, SZ_KV, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(c.dV, SZ_KV, c.hV, SZ_KV, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(c.dSeq, 64, c.hSeq, 64, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(c.dMask, 1024, c.hMask, 1024, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(c.dCfg, 96, c.hCfg, 96, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(c.dOut, SZ_OUT, 0xEE, SZ_OUT);
    aclrtMemset(c.dWsA, SZ_WSACC, 0, SZ_WSACC);
    aclrtMemset(c.dWsM, SZ_WSMS, 0, SZ_WSMS);
    aclrtMemset(c.dWsS, SZ_WSMS, 0, SZ_WSMS);
    aclrtMemset(c.dDbgS, SZ_DBGS, 0, SZ_DBGS);
    aclrtMemset(c.dDbgP, SZ_DBGP, 0, SZ_DBGP);
    aclrtMemset(c.dDbgC, SZ_DBGC, 0, SZ_DBGC);
    aclrtMemset(c.dDbgC2, SZ_DBGC, 0, SZ_DBGC);
    aclrtMemset(c.dGP, SZ_GP, 0, SZ_GP);

    // **负向对照 `negkv`**：调用点把 K/V 两个实参互换（抽出的段一字未改）
    void* kArg = swapKv ? c.dV : c.dK;
    void* vArg = swapKv ? c.dK : c.dV;
    m15_attn_core_kernel<<<numBlocks, 0, c.stream>>>((uint8_t*)c.dQ, (uint8_t*)kArg, (uint8_t*)vArg,
                                                     (uint8_t*)c.dOut, (uint32_t*)c.dSeq, (uint8_t*)c.dWsA,
                                                     (uint8_t*)c.dWsM, (uint8_t*)c.dWsS, (float*)c.dMask,
                                                     (uint8_t*)c.dDbgS, (uint8_t*)c.dDbgP,
                                                     (uint8_t*)c.dDbgC, (uint32_t*)c.dCfg,
                                                     (uint8_t*)c.dDbgC2, (uint8_t*)c.dGP);
    const aclError err = aclrtSynchronizeStream(c.stream);
    if (err != ACL_SUCCESS) {
        printf("[M101][FAIL] seq=%u launch/sync error %d\n", seq, (int)err);
        return 1;
    }
    aclrtMemcpy(c.hOut, SZ_OUT, c.dOut, SZ_OUT, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hWsA, SZ_WSACC, c.dWsA, SZ_WSACC, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hWsM, SZ_WSMS, c.dWsM, SZ_WSMS, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hWsS, SZ_WSMS, c.dWsS, SZ_WSMS, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hDbgS, SZ_DBGS, c.dDbgS, SZ_DBGS, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hDbgP, SZ_DBGP, c.dDbgP, SZ_DBGP, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hDbgC, SZ_DBGC, c.dDbgC, SZ_DBGC, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hDbgC2, SZ_DBGC, c.dDbgC2, SZ_DBGC, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(c.hGP, SZ_GP, c.dGP, SZ_GP, ACL_MEMCPY_DEVICE_TO_HOST);

    if (!dump) return 0;

    char prefix[64];
    if (synth) {
        snprintf(prefix, sizeof(prefix), "m10_synth%u_s%u_", synth, seq);
    } else {
        snprintf(prefix, sizeof(prefix), "m10_case_s%u_", seq);
    }
    char path[128];
    struct Item { const char* suf; const void* p; size_t n; };
    const Item items[] = {
        {"q.bin", c.hQ, SZ_Q},
        {"k.bin", c.hK, (size_t)2 * seqPad * 256 * 2},
        {"v.bin", c.hV, (size_t)2 * seqPad * 256 * 2},
        {"out.bin", c.hOut, SZ_OUT},
        {"wsacc.bin", c.hWsA, SZ_WSACC},
        {"wsm.bin", c.hWsM, SZ_WSMS},
        {"wss.bin", c.hWsS, SZ_WSMS},
        {"dbgs.bin", c.hDbgS, SZ_DBGS},
        {"dbgp.bin", c.hDbgP, SZ_DBGP},
        {"dbgc.bin", c.hDbgC, SZ_DBGC},
        {"dbgc2.bin", c.hDbgC2, SZ_DBGC},
        {"gp.bin", c.hGP, SZ_GP},
    };
    bool ok = true;
    for (const Item& it : items) {
        snprintf(path, sizeof(path), "%s%s", prefix, it.suf);
        ok = DumpBin(path, it.p, it.n) && ok;
    }
    printf("[M101] case seq=%u dumped (%s) %s\n", seq, prefix, ok ? "ok" : "**有写失败**");

    if (refDir != nullptr && *refDir != '\0' && swapKv == false && synth == 0) {
        printf("[M101] 与归档逐字节对拍（refDir=%s，用 `head` 而不是绝对断言）:\n", refDir);
        for (const Item& it : items) {
            char name[96];
            snprintf(name, sizeof(name), "%s%s", prefix, it.suf);
            const long r = CmpWithRef(refDir, name, it.p, it.n);
            if (r == 0) {
                printf("[M101][bincmp] %-28s 逐字节相同（%zu B）\n", name, it.n);
            } else if (r > 0) {
                printf("[M101][bincmp] %-28s **首个不同字节下标 %ld**（共 %zu B）\n", name, r - 1, it.n);
            } else if (r == -1) {
                printf("[M101][bincmp] %-28s 归档里没有这个文件（跳过）\n", name);
            } else {
                printf("[M101][bincmp] %-28s 尺寸/缓冲异常（rc=%ld）\n", name, r);
            }
        }
    }
    return ok ? 0 : 1;
}

// ---- 主流程 ----
static int32_t CoreMain(const char* argvCases, const char* outDir, const char* refDir, bool swapKv)
{
    int64_t aicNum = 0;
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicNum) != ACL_SUCCESS || aicNum <= 0) {
        printf("[M101] aclrtGetDeviceInfo(AICORE_CORE_NUM) failed, fallback 28\n");
        aicNum = 28;
    }
    const uint32_t numBlocks = (uint32_t)aicNum;
    printf("[M101] numBlocks(AIC) = %u\n", numBlocks);
    if (numBlocks != H_SCRATCH_SETUP) {
        printf("[M101][FAIL] 段的 schedule 假定 28 AIC（14 split × 2 KV 头），实得 %u —— 拒绝启动\n",
               numBlocks);
        return 2;
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    const uint32_t casesAll[] = {256, 300, 4096};
    uint32_t casesEnv[4] = {0, 0, 0, 0};
    uint32_t nCases = 3;
    const uint32_t* cases = casesAll;
    if (argvCases != nullptr && *argvCases != '\0') {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s", argvCases);
        nCases = 0;
        for (char* tok = strtok(buf, ","); tok != nullptr && nCases < 4; tok = strtok(nullptr, ",")) {
            char* endp = nullptr;
            const long v = strtol(tok, &endp, 10);
            if (endp == tok || *endp != '\0' || v < 1 || v > (long)H_MAX_SEQ) {
                fprintf(stderr, "[M101] M15AC_CASES 非法项 '%s'：seq 必须为 1..%u —— 拒绝启动\n", tok,
                        H_MAX_SEQ);
                aclrtDestroyStream(stream);
                aclrtResetDevice(0);
                aclFinalize();
                return 1;
            }
            casesEnv[nCases++] = (uint32_t)v;
        }
        if (nCases == 0) {
            fprintf(stderr, "[M101] M15AC_CASES='%s' 解析后为空 —— 拒绝启动\n", argvCases);
            aclrtDestroyStream(stream);
            aclrtResetDevice(0);
            aclFinalize();
            return 1;
        }
        cases = casesEnv;
    }

    if (outDir != nullptr && *outDir != '\0' && chdir(outDir) != 0) {
        fprintf(stderr, "[M101][FAIL] chdir('%s') 失败 —— 检查目录存在\n", outDir);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

    RunCtx c;
    c.stream = stream;
    if (!Alloc(c)) {
        fprintf(stderr, "[M101][FAIL] buffer 分配失败\n");
        return 1;
    }
    ParseL0Cfg(c);
    if (swapKv) printf("[M101] **负向对照 negkv**：调用点把 K/V 实参互换（段一字未改）\n");

    int rc = 0;
    for (uint32_t ci = 0; ci < nCases; ++ci) {
        rc |= RunCoreCase(c, numBlocks, cases[ci], 0u, refDir, swapKv, true);
    }
    printf("[M101] host done; 判据 = donor 的 m10_attn_decode/check_ref.py（本目录原样跑）\n");

    Free(c);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}

// ============================================================
// M101 第四步：o_proj + `×sigmoid(gate)` 的独立判据（对 host fp64 参考）+ 负向对照
// ============================================================
// 判据与 ε 的**每一项出处**（官方规格表，硬件列）：
//   `/workspace/asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md`
//     基础算术 | Exp  | 1ulp, not support denormalized numbers
//     基础算术 | Adds | half/float: 0ulp   （⇒ 精确）
//     基础算术 | Mul  | half/float: 0ulp   （⇒ 精确）
//     基础算术 | Div  | 1ulp, not support denormalized numbers
//   u = 2^-24（fp32 单位舍入）。σ(g) = 1/(1+exp(−g)) 的误差账（**最坏情形换算，不是实测**）：
//     Muls(g,−1) 精确（2 的幂）; Exp ≤1 ulp ⇒ 相对 ≤2u; Adds(+1) 精确; Div ≤1 ulp(结果) ⇒ 相对 ≤2u;
//     分母的 2u 传到 σ 上再乘 e^{−g}/(1+e^{−g}) ≤ 1 ⇒ 再加 2u   ⇒ |Δσ/σ| ≤ **4u**
//   t = a·σ：Mul 0 ulp ⇒ t 的相对误差完全来自 σ。bf16→fp32 Cast 精确。于是设备在 cast 前的 fp32 值
//   `v` 满足 |v − t_ref| ≤ 4u·|t_ref|（t_ref = fp64 的 a·σ），RNE 到 bf16 再加**半步**量化 ⇒
//   判据 OP-B：`|f32(t_dev) − t_ref| ≤ 0.5·ulp_bf16(t_ref) + 4u·|t_ref|`
//   **分辨率声明**：该判据对 t 的相对误差的分辨力 ≈ 0.5 bf16 半步 ≈ 0.39%（bf16 尾数 7 位）；
//   更细的 σ 误差**不在此判据的可判范围内**（与 M10 README 判据 E 的同款写法）。
// 判据 OP-A（GEMM，逐位）：输入取**精确整数域**（t、W ∈ [-8,8] 整数；乘积 ≤64；K=6144 ⇒ |Σ| ≤ 393216 < 2^24
//   ⇒ fp32 任何累加次序都精确）⇒ 设备 bf16 位型必须与 fp64 参考 RNE 后**逐位相同**（donor m11 已实证的判据）。
// 判据 OP-C（e2e，**S1 定位声明**）：输入 = **设备产出的 t**（D2H 回读），参考 = fp64 Σ W_j t_j，
//   tol = K·u·Σ|W_j t_j|（mmad k=6144 的 fp32 累加保守界，**推导**）。**被判量** = "GEMM 消费的 t
//   是否是门控段产出的那一份（同布局/同顺序）"；**t 本身不是本判据的被判量**（两侧同一份设备 t），
//   t 由判据 OP-B 咬。
namespace opd {

constexpr uint32_t OP_M = 2560;
constexpr uint32_t OP_KK = 6144;
constexpr double U24 = 2.0 / 16777216.0;                // 2^-24

static double Bf16ToDouble(uint16_t h) { return (double)Bf16ToFloat(h); }
static double SigmoidF64(double g) { return 1.0 / (1.0 + exp(-g)); }

static float HashFrac(uint32_t i, uint32_t salt)
{
    return (float)(Hash3(i, 7u, salt) & 0xFFFFu) / 65535.0f * 1.6f - 0.8f;
}

static void GenFracBf16(uint16_t* bits, uint32_t count, uint32_t salt)
{
    for (uint32_t i = 0; i < count; ++i) bits[i] = FloatToBf16(HashFrac(i, salt));
}

static void GenIntBf16(uint16_t* bits, uint32_t count, uint32_t salt)
{
    for (uint32_t i = 0; i < count; ++i) {
        const int v = (int)(Hash3(i, 13u, salt) % 17u) - 8;
        bits[i] = FloatToBf16((float)v);
    }
}

// bf16 的**格点间距** = 2^(e-7)，其中 |v| ∈ [2^(e-1), 2^e)（frexp 的 e 即该区间的上界指数）。
// 为什么不用「取位型相邻值再做差」的写法（本 M101 的一次实测教训，如实记账）：
//   位型的"相邻"在 bf16（符号-幅值）里对**负数**是**更靠近 0** 的那个邻居（0xBE80 → 0xBE7F = −0.25 → −0.249023），
//   于是拿它当"1 个 bf16 ulp"会得到只有正数侧一半的值。M10 的判据只用在 P̃ ≥ 0 上，碰不到这个分支；
//   本判据的 t 有正有负 ⇒ 必须按 |v| 的 binade 直接算间距。**第一次跑出的 OP-B FAIL 就是这个写法导致的
//   假 FAIL（设备侧 6144/6144 逐位等于参考量化值）**，见 README §4.3。
static double Bf16BinadeSpacing(double v)
{
    const double a = v < 0 ? -v : v;
    if (a == 0.0) return 0.0;
    int e = 0;
    (void)frexp(a, &e);
    return ldexp(1.0, e - 8);
}

// 判据 OP-B 的量化项：RNE 到 bf16 的最大量化误差 = 所在 binade 的**半步**。
// 参考（未量化）与设备值（已量化）可能分处相邻 binade 的两侧 ⇒ 取两者间距的较大者（保守上界）。
static double Bf16QuantHalf(double ref, double got)
{
    const double s1 = Bf16BinadeSpacing(ref);
    const double s2 = Bf16BinadeSpacing(got);
    return 0.5 * (s1 > s2 ? s1 : s2);
}

struct OpTally {
    uint32_t n = 0;
    uint32_t bad = 0;
    double worstRatio = 0.0;    // 最大"界占用"（越界程度）
    double maxAbs = 0.0;
    double maxRel = 0.0;
    uint32_t d0 = 0, d1 = 0, d2 = 0;   // bf16 位差分布（0 / 1 / ≥2 个格点）
};

// ---- 判据 OP-A：GEMM 逐位（t/W 都在精确整数域）----
static OpTally CheckGemmBitExact(const uint16_t* tHost, const uint16_t* wHost, const uint16_t* yHost,
                                uint32_t m)
{
    OpTally T;
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < OP_M; ++j) {
            double acc = 0.0;
            const uint16_t* wRow = wHost + (size_t)j * OP_KK;
            for (uint32_t kk = 0; kk < OP_KK; ++kk) {
                acc += Bf16ToDouble(tHost[(size_t)i * OP_KK + kk]) * Bf16ToDouble(wRow[kk]);
            }
            const uint16_t expect = FloatToBf16((float)acc);   // |acc| ≤ 393216 < 2^24 ⇒ 无双重舍入
            const uint16_t got = yHost[(size_t)i * OP_M + j];
            ++T.n;
            if (got != expect) {
                if (T.bad < 6) {
                    printf("      OP-A 位不等: y[%u][%u] got 0x%04x (%g) expect 0x%04x (%g)\n", i, j, got,
                           Bf16ToDouble(got), expect, Bf16ToDouble(expect));
                }
                ++T.bad;
            }
        }
    }
    return T;
}

// ---- 判据 OP-B / OP-C：容差式 ----
// ck = 0: OP-B（t_dev vs t_ref，量化半步 + 4u 相对）
// ck = 1: OP-C（y_dev vs fp64 Σ W·t_dev，mmad k 累加界）
static OpTally CheckGateTol(const uint16_t* attn, const uint16_t* gate, const uint16_t* tDev, uint32_t m)
{
    OpTally T;
    for (uint32_t i = 0; i < m * OP_KK; ++i) {
        const double a = Bf16ToDouble(attn[i]);
        const double g = Bf16ToDouble(gate[i]);
        const double ref = a * SigmoidF64(g);
        const double got = Bf16ToDouble(tDev[i]);
        const double diff = got - ref;
        const double adiff = diff < 0 ? -diff : diff;
        const double half = Bf16QuantHalf(ref, got);
        const double tol = half + 4.0 * U24 * (ref < 0 ? -ref : ref);   // 4u = Exp 2u + Div 2u（§4.2）
        ++T.n;
        T.maxAbs = adiff > T.maxAbs ? adiff : T.maxAbs;
        if (ref != 0.0) {
            const double rel = adiff / (ref < 0 ? -ref : ref);
            T.maxRel = rel > T.maxRel ? rel : T.maxRel;
        }
        if (adiff > tol) {
            ++T.bad;
            if (T.bad < 6) {
                printf("      OP-B 越界: t[%u] got %g ref %g |Δ|=%g tol=%g\n", i, got, ref, adiff, tol);
            }
        }
        if (half > 0.0) {
            const double r = (adiff - 4.0 * U24 * (ref < 0 ? -ref : ref)) / half;
            T.worstRatio = r > T.worstRatio ? r : T.worstRatio;
        }
        // 位差分布（对 bf16 网格化后的参考）
        const uint16_t rb = FloatToBf16((float)ref);
        const int32_t d = (int32_t)(tDev[i] & 0x7FFF) - (int32_t)(rb & 0x7FFF);
        const int32_t ad = d < 0 ? -d : d;
        if ((tDev[i] & 0x8000) != (rb & 0x8000)) {
            ++T.d2;
        } else if (ad == 0) {
            ++T.d0;
        } else if (ad == 1) {
            ++T.d1;
        } else {
            ++T.d2;
        }
    }
    return T;
}

static OpTally CheckGemmTol(const uint16_t* tDev, const uint16_t* wHost, const uint16_t* yHost, uint32_t m)
{
    OpTally T;
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < OP_M; ++j) {
            double acc = 0.0;
            double absSum = 0.0;
            const uint16_t* wRow = wHost + (size_t)j * OP_KK;
            for (uint32_t kk = 0; kk < OP_KK; ++kk) {
                const double wv = Bf16ToDouble(wRow[kk]);
                const double tv = Bf16ToDouble(tDev[(size_t)i * OP_KK + kk]);
                acc += wv * tv;
                absSum += (wv < 0 ? -wv : wv) * (tv < 0 ? -tv : tv);
            }
            const double tol = (double)OP_KK * U24 * absSum;   // mmad k 累加的保守界（推导）
            const double got = Bf16ToDouble(yHost[(size_t)i * OP_M + j]);
            const double diff = got - acc;
            const double adiff = diff < 0 ? -diff : diff;
            ++T.n;
            T.maxAbs = adiff > T.maxAbs ? adiff : T.maxAbs;
            if (adiff > tol) {
                ++T.bad;
                if (T.bad < 6) {
                    printf("      OP-C 越界: y[%u][%u] got %g ref %g |Δ|=%g tol=%g\n", i, j, got, acc, adiff,
                           tol);
                }
            }
            if (tol > 0.0) {
                const double r = adiff / tol;
                T.worstRatio = r > T.worstRatio ? r : T.worstRatio;
            }
        }
    }
    return T;
}

static int LaunchGate(aclrtStream st, void* attn, void* gate, void* t, uint32_t m, uint32_t mode,
                      uint32_t blockDim, const char* tag)
{
    m15_attn_gate_kernel<<<blockDim, 0, st>>>((uint8_t*)attn, (uint8_t*)gate, (uint8_t*)t, m, mode);
    const aclError e = aclrtSynchronizeStream(st);
    if (e != ACL_SUCCESS) {
        printf("[M101][FAIL] %s launch/sync error %d\n", tag, (int)e);
        return 1;
    }
    return 0;
}

static int LaunchGemm(aclrtStream st, void* t, void* w, void* y, uint32_t m, uint32_t mode, const char* tag)
{
    m15_attn_oproj_kernel<<<1, 0, st>>>((uint8_t*)t, (uint8_t*)w, (uint8_t*)y, m, mode);
    const aclError e = aclrtSynchronizeStream(st);
    if (e != ACL_SUCCESS) {
        printf("[M101][FAIL] %s launch/sync error %d\n", tag, (int)e);
        return 1;
    }
    return 0;
}

}  // namespace opd

static int32_t OProjMain(const char* argvCases, const char* outDir)
{
    using namespace opd;
    int64_t aicNum = 0;
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicNum) != ACL_SUCCESS || aicNum <= 0) {
        printf("[M101] aclrtGetDeviceInfo(AICORE_CORE_NUM) failed, fallback 28\n");
        aicNum = 28;
    }
    const uint32_t numBlocks = (uint32_t)aicNum;
    printf("[M101] o_proj: numBlocks(AIC) = %u（门控段用 mix(1,2) + blockDim=%u ⇒ AIV=%u）\n", numBlocks,
           numBlocks, numBlocks * 2u);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    if (outDir != nullptr && *outDir != '\0' && chdir(outDir) != 0) {
        fprintf(stderr, "[M101][FAIL] chdir('%s') 失败\n", outDir);
        return 1;
    }

    uint32_t ms[3] = {1, 3, 0};
    uint32_t nM = 2;
    if (argvCases != nullptr && *argvCases != '\0') {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s", argvCases);
        nM = 0;
        for (char* tok = strtok(buf, ","); tok != nullptr && nM < 3; tok = strtok(nullptr, ",")) {
            char* endp = nullptr;
            const long v = strtol(tok, &endp, 10);
            if (endp == tok || *endp != '\0' || v < 1 || v > 64) {
                fprintf(stderr, "[M101] M15AC_CASES（oproj 档）= m 列表，必须为 1..64，实得 '%s'\n", tok);
                return 1;
            }
            ms[nM++] = (uint32_t)v;
        }
    }
    if (nM == 0) nM = 2;

    const size_t wBytes = (size_t)OP_M * OP_KK * 2;
    const size_t maxMRows = 4u;                       // kernel 侧 m<2 提升为 2 行、m=3 用 calcM=3
    const size_t tBytes = maxMRows * OP_KK * 2;
    const size_t yBytes = maxMRows * OP_M * 2;

    void *dAttn = nullptr, *dGate = nullptr, *dT = nullptr, *dW = nullptr, *dY = nullptr;
    aclrtMalloc(&dAttn, tBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dGate, tBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dT, tBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dW, wBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    uint16_t* hAttn = (uint16_t*)malloc(tBytes);
    uint16_t* hGate = (uint16_t*)malloc(tBytes);
    uint16_t* hT = (uint16_t*)malloc(tBytes);
    uint16_t* hTint = (uint16_t*)malloc(tBytes);
    uint16_t* hW = (uint16_t*)malloc(wBytes);
    uint16_t* hY = (uint16_t*)malloc(yBytes);

    GenIntBf16(hW, (uint32_t)(wBytes / 2), 0x5A01u);
    GenFracBf16(hAttn, (uint32_t)(tBytes / 2), 0xA101u);
    GenFracBf16(hGate, (uint32_t)(tBytes / 2), 0xA102u);
    GenIntBf16(hTint, (uint32_t)(tBytes / 2), 0xA103u);
    aclrtMemcpy(dW, wBytes, hW, wBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dAttn, tBytes, hAttn, tBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dGate, tBytes, hGate, tBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    uint32_t nJudge = 0, nJudgeFail = 0;
    int rc = 0;

    for (uint32_t ci = 0; ci < nM; ++ci) {
        const uint32_t m = ms[ci];
        printf("[op] m=%u ====\n", m);
        char path[128];
        // 输入张量的指纹（attn/gate/tint 落盘；W 31.5MB 不落盘但给指纹）——
        // 离线复核脚本据此证明"我按公式重生成的输入与本次运行是同一批字节"（复审 F1）
        printf("[M101][fp] %-20s file=%-28s len=%-9zu fnv1a64=%016llx\n", "input W (not dumped)", "-",
               wBytes, (unsigned long long)Fnv1a64(hW, wBytes));
        snprintf(path, sizeof(path), "m101op_m%u_attn.bin", m);
        DumpAndPrint("input attn", path, hAttn, (size_t)m * OP_KK * 2);
        snprintf(path, sizeof(path), "m101op_m%u_gate.bin", m);
        DumpAndPrint("input gate", path, hGate, (size_t)m * OP_KK * 2);
        snprintf(path, sizeof(path), "m101op_m%u_tint.bin", m);
        DumpAndPrint("input t (integer)", path, hTint, (size_t)m * OP_KK * 2);

        // ---- 判据 OP-A：GEMM 逐位（精确整数域），契约档 + 负向档 ----
        aclrtMemcpy(dT, tBytes, hTint, tBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (LaunchGemm(stream, dT, dW, dY, m, M15OP::GEMM_MODE_CONTRACT, "OP-A contract")) return 1;
        aclrtMemcpy(hY, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OpTally A = CheckGemmBitExact(hTint, hW, hY, m);
        ++nJudge;
        if (A.bad) ++nJudgeFail;
        printf("    判定项[OP-A. o_proj GEMM 逐位（精确整数域）]: %s 比较 %u 元素、位不等 %u\n",
               A.bad ? "FAIL" : "PASS", A.n, A.bad);
        // **契约档的 y 必须在这里就落盘**（复审 r1-F1 的成因：旧版在循环末尾统一 dump，
        //  那时 hY 已被 OP-C 的 KMINUS1 负向档覆盖 ⇒ 落下去的是"故意弄坏"的输出）
        snprintf(path, sizeof(path), "m101op_m%u_yA_contract.bin", m);
        DumpAndPrint("OP-A y (contract)", path, hY, (size_t)m * OP_M * 2);

        if (LaunchGemm(stream, dT, dW, dY, m, M15OP::GEMM_MODE_KMINUS1, "OP-A neg k-1")) return 1;
        aclrtMemcpy(hY, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OpTally Aneg = CheckGemmBitExact(hTint, hW, hY, m);
        printf("    负向[OP-A · GEMM_MODE_KMINUS1 少累加一个 baseK 块]: 位不等 %u/%u（**必须 > 0**）\n",
               Aneg.bad, Aneg.n);
        if (Aneg.bad == 0) ++rc;
        snprintf(path, sizeof(path), "m101op_m%u_yA_kminus1.bin", m);
        DumpAndPrint("OP-A y (K-1 neg)", path, hY, (size_t)m * OP_M * 2);

        // ---- 判据 OP-B：×sigmoid(gate)（容差式），契约档 + 两个负向档 ----
        if (LaunchGate(stream, dAttn, dGate, dT, m, M15OP::GATE_MODE_CONTRACT, numBlocks, "OP-B contract"))
            return 1;
        aclrtMemcpy(hT, tBytes, dT, tBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OpTally B = CheckGateTol(hAttn, hGate, hT, m);
        ++nJudge;
        if (B.bad) ++nJudgeFail;
        printf("    判定项[OP-B. ×sigmoid(gate)（T3，量化半步 + 4u 相对）]: %s 比较 %u 元素、越界 %u "
               "| maxAbsErr %g | 最大界占用 %.4f | 位差 0/1/≥2 格点 = %u/%u/%u\n",
               B.bad ? "FAIL" : "PASS", B.n, B.bad, B.maxAbs, B.worstRatio, B.d0, B.d1, B.d2);
        // guard：判别力对照
        //   ① 把"参考按 RNE 量化到 bf16"当设备 ⇒ 越界必须 0（验证容差 ≥ 纯量化半步，判据不空洞）
        //   ② 把设备值的前 8 个元素再挪 +2 个 bf16 格点 ⇒ 越界必须 > 0（验证判据有分辨力）
        {
            uint16_t* hMoved = (uint16_t*)malloc(tBytes);
            for (uint32_t i = 0; i < m * OP_KK; ++i) {
                const double a = Bf16ToDouble(hAttn[i]);
                const double g = Bf16ToDouble(hGate[i]);
                hMoved[i] = FloatToBf16((float)(a * SigmoidF64(g)));
            }
            OpTally g0 = CheckGateTol(hAttn, hGate, hMoved, m);
            memcpy(hMoved, hT, tBytes);
            uint32_t nUp = 0;
            for (uint32_t i = 0; i < m * OP_KK && nUp < 8; ++i) {
                const uint16_t b = hMoved[i];
                // bf16 是符号-幅值：**幅值 +2** 才是"再远 2 个格点"（两种符号都是加）；
                // 幅值封顶在 0x7F7F（bf16 的最大有限值），避免越到 inf/NaN
                const uint32_t mag = (uint32_t)(b & 0x7FFFu);
                const uint32_t nmag = (mag + 2u > 0x7F7Fu) ? 0x7F7Fu : (mag + 2u);
                hMoved[i] = (uint16_t)((uint32_t)(b & 0x8000u) | nmag);
                ++nUp;
            }
            OpTally g2 = CheckGateTol(hAttn, hGate, hMoved, m);
            printf("    guard[OP-B 判别力对照]: 参考量化到 bf16 当设备 ⇒ 越界 %u（须 0）；前 %u 个元素 +2 格点 ⇒ 越界 %u（须 >0）\n",
                   g0.bad, nUp, g2.bad);
            if (g0.bad != 0 || g2.bad == 0) ++rc;
            free(hMoved);
        }
        const uint32_t negModes[2] = {M15OP::GATE_MODE_NO_GATE, M15OP::GATE_MODE_SIGN};
        for (uint32_t mi = 0; mi < 2u; ++mi) {
            const uint32_t mode = negModes[mi];
            if (LaunchGate(stream, dAttn, dGate, dT, m, mode, numBlocks, "OP-B neg")) return 1;
            aclrtMemcpy(hT, tBytes, dT, tBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            OpTally Bn = CheckGateTol(hAttn, hGate, hT, m);
            printf("    负向[OP-B · %s]: 越界 %u/%u（**必须 > 0**）、最大界占用 %.3f\n",
                   mode == M15OP::GATE_MODE_NO_GATE ? "GATE_MODE_NO_GATE 跳过门控" : "GATE_MODE_SIGN σ(−g)",
                   Bn.bad, Bn.n, Bn.worstRatio);
            if (Bn.bad == 0) ++rc;
        }
        // 恢复设备 t（契约档）供 OP-C 用
        if (LaunchGate(stream, dAttn, dGate, dT, m, M15OP::GATE_MODE_CONTRACT, numBlocks, "OP-C ref t"))
            return 1;
        aclrtMemcpy(hT, tBytes, dT, tBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        snprintf(path, sizeof(path), "m101op_m%u_tdev.bin", m);
        DumpAndPrint("OP-B t (contract)", path, hT, (size_t)m * OP_KK * 2);

        // ---- 判据 OP-C：e2e（S1：输入 = 设备 t），契约档 + 负向档 ----
        if (LaunchGemm(stream, dT, dW, dY, m, M15OP::GEMM_MODE_CONTRACT, "OP-C contract")) return 1;
        aclrtMemcpy(hY, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OpTally C = CheckGemmTol(hT, hW, hY, m);
        ++nJudge;
        if (C.bad) ++nJudgeFail;
        printf("    判定项[OP-C. e2e（S1：输入 = 设备 t）]: %s 比较 %u 元素、越界 %u | maxAbsErr %g | "
               "最大界占用 %.4f\n",
               C.bad ? "FAIL" : "PASS", C.n, C.bad, C.maxAbs, C.worstRatio);
        snprintf(path, sizeof(path), "m101op_m%u_yC_contract.bin", m);
        DumpAndPrint("OP-C y (contract)", path, hY, (size_t)m * OP_M * 2);
        if (LaunchGemm(stream, dT, dW, dY, m, M15OP::GEMM_MODE_KMINUS1, "OP-C neg")) return 1;
        aclrtMemcpy(hY, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OpTally Cneg = CheckGemmTol(hT, hW, hY, m);
        printf("    负向[OP-C · GEMM_MODE_KMINUS1]: 越界 %u/%u（**必须 > 0**）\n", Cneg.bad, Cneg.n);
        if (Cneg.bad == 0) ++rc;
        snprintf(path, sizeof(path), "m101op_m%u_yC_kminus1.bin", m);
        DumpAndPrint("OP-C y (K-1 neg)", path, hY, (size_t)m * OP_M * 2);
    }

    printf("[M101] o_proj：判定项 %u/%u 通过%s（另：负向/guard 任一项未变红则退出码非 0）\n",
           nJudge - nJudgeFail, nJudge, (nJudgeFail == 0 && rc == 0) ? "" : " —— 有 FAIL");

    free(hAttn); free(hGate); free(hT); free(hTint); free(hW); free(hY);
    aclrtFree(dAttn); aclrtFree(dGate); aclrtFree(dT); aclrtFree(dW); aclrtFree(dY);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return (nJudgeFail == 0 && rc == 0) ? 0 : 1;
}

}  // namespace detail
}  // namespace M15ACHost

#endif  // M15_ATTN_CORE_HOST_H
