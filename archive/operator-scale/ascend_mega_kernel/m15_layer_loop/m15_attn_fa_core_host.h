// ============================================================
// m15_attn_fa_core_host.h —— M117 / Wave-B3 的 **host 侧**（数据生成 + 落盘 + 启动）
//
// 位置说明：按 `docs/15` §M103-2.2 的 B3 行，host 头与段体头同在 `m15_layer_loop/`；
// 独立验证路 `m25_attn_fa_core/` 的 `.asc` 通过 `-I ../m15_layer_loop` 引用它（**不复制**）。
//
// 本文件只被 `m25_attn_fa_core/m25_attn_fa_core.asc` 的 `main()` 使用（**不**被段体头依赖）。
// 它做三件事：
//   ① 用**确定性**生成器造 Q/K/V（bf16），写进 GM 的 Q 平面 / 主 KV 平面；
//   ② 启动 `M15FAC::FaCoreBody`；
//   ③ 把**实际用到的** q/k/v（契约语义：`[row][head][dim]`）与输出落盘，供 numpy fp64 参考对拍。
//
// **不写绝对断言**：所有判据都在 `check_ref.py` 里；本文件只产出数据 + 打印读数。
// 负向对照（把被测对象弄坏）：
//   `negmask`  -> 调 `FaCoreBody<false>`（**关掉三角掩码**）
//   `negshift` -> KV 平面指针整体后移 16 个 token（K/V 与位置错位）
//   `negstart` -> 传 `posBase + 1`（因果窗口整体错位）
// 三者的 `params.txt` 都写**正确**的 posBase，于是参考是正确的那份 ⇒ 判据必须变红。
// ============================================================
#ifndef M15_ATTN_FA_CORE_HOST_H
#define M15_ATTN_FA_CORE_HOST_H

#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace M15FAHost {
namespace detail {

constexpr uint32_t H_NH = M15FAC::FAC_NH;
constexpr uint32_t H_HD = M15FAC::FAC_HD;
constexpr uint32_t H_DV = M15FAC::FAC_DV;
constexpr uint32_t H_NKV = M15FAC::FAC_NKV;

// Q 平面行距 = `m15_attn_prolog.h::AP_OUT_N`（**对齐 prolog 产物，不另造布局**）
constexpr uint32_t H_Q_STRIDE = 13952;

constexpr uint32_t H_BLK_BYTES = M15Kv::KV_BLOCK_BYTES;
constexpr uint32_t H_BLK_TOK = M15Kv::KV_BLOCK_TOKENS;
constexpr uint32_t H_SIN = M15FAC::FAC_SIN;
constexpr uint32_t H_P = M15FAC::FAC_P;
constexpr size_t kWitElems = 56u * 32u * 2u;                 // AIC 28 + AIV 56 槽 × 32 id × 2 种
constexpr size_t kWitBytes = kWitElems * sizeof(int32_t);

inline float Bf16ToF(uint16_t b)
{
    const uint32_t x = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// ---- 确定性生成器（host 与任何复算方都能重跑出同一批字节）----
inline uint32_t Hash3(uint32_t i, uint32_t j, uint32_t salt)
{
    uint32_t x = i * 2654435761u ^ j * 2246822519u ^ salt * 3266489917u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return x;
}

// 造一个 bf16：|v| ∈ [0.5, 1.0)，符号随机（⇒ 点积有正有负，softmax 有真实分布）
inline uint16_t GenBf16(uint32_t i, uint32_t j, uint32_t salt)
{
    const uint32_t r = Hash3(i, j, salt);
    const uint32_t mant = r & 0x7Fu;                  // 7 bit 尾数
    const uint32_t sign = (r >> 7) & 1u;
    const uint32_t expn = 126u;                       // 2^-1 → [0.5, 1.0)
    return (uint16_t)((sign << 15) | (expn << 7) | mant);
}

inline bool EnsureDir(const char* dir)
{
    struct stat st;
    if (stat(dir, &st) == 0) { return S_ISDIR(st.st_mode); }
    // 逐级创建父目录：`run_checks.sh` 的 `$OUT/rep<i>/` 与 `argv[2]` 的多层路径需要它。
    std::string s(dir);
    while (s.size() > 1 && s.back() == '/') { s.pop_back(); }
    for (size_t i = 1; i <= s.size(); ++i) {
        if (i != s.size() && s[i] != '/') { continue; }
        const std::string sub = s.substr(0, i);
        if (sub.empty()) { continue; }
        if (stat(sub.c_str(), &st) == 0) {
            if (!S_ISDIR(st.st_mode)) { return false; }
        } else if (mkdir(sub.c_str(), 0755) != 0) {
            return false;
        }
    }
    return true;
}

inline bool WriteFile(const std::string& path, const void* data, size_t bytes)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) { return false; }
    const size_t n = fwrite(data, 1, bytes, f);
    fclose(f);
    return n == bytes;
}

inline std::string StripTrailingSlash(const char* d)
{
    std::string s(d == nullptr ? "" : d);
    while (s.size() > 1 && s.back() == '/') { s.pop_back(); }
    return s;
}

struct LaunchArgs {
    uint32_t m;
    uint32_t posBase;
    uint32_t ctx;
    uint32_t lane;
    uint32_t nBlk;
};

// ---- 一份测试（数据生成 → H2D → 启动 → D2H → 落盘）----
inline int RunOne(const char* kernelName, const char* outDir, const char* mode, const LaunchArgs& a,
                  const std::string& paramsText)
{
    const uint32_t m = a.m;
    const uint32_t ctx = a.ctx;
    const uint32_t rowsPadded = ((m + H_P - 1u) / H_P) * H_P;
    const uint32_t blocks = (ctx + H_BLK_TOK - 1u) / H_BLK_TOK;

    const size_t qFloats = (size_t)rowsPadded * H_Q_STRIDE;
    const size_t qBytes = qFloats * 2u;
    const size_t kvBytes = (size_t)blocks * H_BLK_BYTES;
    const size_t outFloats = (size_t)rowsPadded * H_NH * H_HD;   // out 平面按 FAC_P 行圆整
    const size_t outBytes = outFloats * 2u;
    const size_t pBytes = (size_t)a.nBlk * 2u * H_P * H_SIN * 2u;

    std::vector<uint16_t> hQ(qFloats, 0);
    std::vector<uint8_t> hKV(kvBytes, 0);
    std::vector<uint16_t> hOut(outFloats, 0);
    // 契约语义下的 q/k/v（供参考对拍）
    std::vector<uint16_t> qSem((size_t)m * H_NH * H_HD, 0);
    std::vector<uint16_t> kSem((size_t)ctx * H_NKV * H_HD, 0);
    std::vector<uint16_t> vSem((size_t)ctx * H_NKV * H_HD, 0);

    // Q 平面：头 h 占列 [h*HD, h*HD+HD)；**语义平面**同步一份
    for (uint32_t r = 0; r < m; ++r) {
        for (uint32_t h = 0; h < H_NH; ++h) {
            for (uint32_t d = 0; d < H_HD; ++d) {
                const uint16_t v = GenBf16(r, h * H_HD + d, 101u);
                hQ[(size_t)r * H_Q_STRIDE + (size_t)h * H_HD + d] = v;
                qSem[((size_t)r * H_NH + h) * H_HD + d] = v;
            }
        }
    }
    // 主 KV 平面：只用 `m15_attn_kv.h` 的寻址宏（不自带第二份数字）
    uint8_t* kvp = hKV.data();
    for (uint32_t pos = 0; pos < ctx; ++pos) {
        for (uint32_t n2 = 0; n2 < H_NKV; ++n2) {
            for (uint32_t d = 0; d < H_HD; ++d) {
                const uint16_t kv = GenBf16(pos, n2 * H_HD + d, 202u);
                const uint16_t vv = GenBf16(pos, n2 * H_HD + d, 303u);
                const uint64_t offK = (uint64_t)M15KV_KV_BYTE_OFF_CONTIG(pos, n2, M15Kv::KV_LANE_K, d);
                const uint64_t offV = (uint64_t)M15KV_KV_BYTE_OFF_CONTIG(pos, n2, M15Kv::KV_LANE_V, d);
                memcpy(kvp + offK, &kv, 2);
                memcpy(kvp + offV, &vv, 2);
                kSem[((size_t)pos * H_NKV + n2) * H_HD + d] = kv;
                vSem[((size_t)pos * H_NKV + n2) * H_HD + d] = vv;
            }
        }
    }

    // ---- 设备侧 ----
    void* dQ = nullptr;
    void* dKV = nullptr;
    void* dOut = nullptr;
    void* dP = nullptr;
    void* dWit = nullptr;
    if (aclrtMalloc(&dQ, qBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dKV, kvBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dP, pBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&dWit, kWitBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        printf("[M25FA][FAIL] aclrtMalloc\n");
        return 40;
    }
    aclrtMemset(dOut, outBytes, 0xEE, outBytes);
#if defined(M25FA_AICDMA)
    aclrtMemset(dP, pBytes, 0x5A, pBytes);   // 探针：预填已知图案，供 AIC 的 GM->UB->GM 往返核对
#else
    aclrtMemset(dP, pBytes, 0, pBytes);
#endif
    aclrtMemset(dWit, kWitBytes, 0, kWitBytes);
    aclrtMemcpy(dQ, qBytes, hQ.data(), qBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dKV, kvBytes, hKV.data(), kvBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // ---- 负向对照的注入点（**只动调用点，不动被测的段体**）----
    uint8_t* kvArg = (uint8_t*)dKV;
    uint32_t posBaseArg = a.posBase;
    bool useNegMask = false;
    if (strcmp(mode, "negmask") == 0) {
        useNegMask = true;
    } else if (strcmp(mode, "negshift") == 0) {
        kvArg = (uint8_t*)dKV + H_BLK_BYTES;          // K/V 与位置错位 16 个 token
    } else if (strcmp(mode, "negstart") == 0) {
        posBaseArg = a.posBase + 1u;                  // 因果窗口整体错位
    }

    printf("[M25FA] pre-launch m=%u posBase=%u ctx=%u nBlk=%u\n", m, posBaseArg, ctx, a.nBlk);
    fflush(stdout);
    if (useNegMask) {
        m25_fa_negmask_kernel<<<a.nBlk, 0, stream>>>((uint8_t*)dQ, H_Q_STRIDE, kvArg, (uint8_t*)dOut,
                                                     (uint8_t*)dP, (uint8_t*)dWit, m, posBaseArg, ctx, a.lane,
                                                     a.nBlk);
    } else {
        m25_fa_kernel<<<a.nBlk, 0, stream>>>((uint8_t*)dQ, H_Q_STRIDE, kvArg, (uint8_t*)dOut, (uint8_t*)dP,
                                             (uint8_t*)dWit, m, posBaseArg, ctx, a.lane, a.nBlk);
    }
    printf("[M25FA] launched, waiting sync\n");
    fflush(stdout);
    // ---- 跨核记账见证的**挂死读回**（M25FA_WITDRAIN=1）：kernel 跑不通时用另一条 stream 把计数拉回来 ----
    if (getenv("M25FA_WITDRAIN") != nullptr) {
        sleep(6);
        std::vector<int32_t> wit(kWitElems, 0);
        aclrtStream s2 = nullptr;
        aclrtCreateStream(&s2);
        const aclError e2 = aclrtMemcpyAsync(wit.data(), kWitBytes, dWit, kWitBytes, ACL_MEMCPY_DEVICE_TO_HOST, s2);
        const aclError e3 = (e2 == ACL_SUCCESS) ? aclrtSynchronizeStream(s2) : e2;
        printf("[M25FA][WIT] drain copy err=%d sync err=%d\n", (int)e2, (int)e3);
        // 只打印非零项： slot*32*2 + id*2 + kind
        int printed = 0;
        for (uint32_t slot = 0; slot < 56u; ++slot) {
            for (uint32_t id = 0; id < 32u; ++id) {
                const int32_t ns = wit[(slot * 32u + id) * 2u + 0u];
                const int32_t nw = wit[(slot * 32u + id) * 2u + 1u];
                if (ns != 0 || nw != 0) {
                    printf("[M25FA][WIT] slot=%u(%s%u) id=%u set=%d wait=%d\n", slot,
                           slot < 28u ? "AIC" : "AIV", slot < 28u ? slot : slot - 28u, id, (int)ns, (int)nw);
                    ++printed;
                }
            }
        }
        printf("[M25FA][WIT] nonzero entries=%d\n", printed);
#if defined(M25FA_AICDMA)
        {
            const uint8_t* bp = (const uint8_t*)wit.data();
            int hits = 0;
            for (int i = 0; i < 1024; ++i) { if (bp[4096 + i] == 0x5A) { ++hits; } }
            printf("[M25FA][AICDMA] wit[4096..5120) == 0x5A 的字节数 = %d / 1024\n", hits);
        }
#endif
        fflush(stdout);
        _exit(0);
    }
    const aclError err = aclrtSynchronizeStream(stream);
    printf("[M25FA] sync done err=%d\n", (int)err);
    fflush(stdout);
    if (err != ACL_SUCCESS) {
        printf("[M25FA][FAIL] launch %s stream sync err=%d\n", kernelName, (int)err);
        return 41;
    }
    aclrtMemcpy(hOut.data(), outBytes, dOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // ---- 落盘 ----
    if (!EnsureDir(outDir)) {
        printf("[M25FA][FAIL] mkdir %s\n", outDir);
        return 42;
    }
    const std::string dir = StripTrailingSlash(outDir);
    const bool ok = WriteFile(dir + "/q.bin", qSem.data(), qSem.size() * 2u) &&
                    WriteFile(dir + "/k.bin", kSem.data(), kSem.size() * 2u) &&
                    WriteFile(dir + "/v.bin", vSem.data(), vSem.size() * 2u) &&
                    WriteFile(dir + "/out.bin", hOut.data(), (size_t)m * H_NH * H_HD * 2u) &&
                    WriteFile(dir + "/params.txt", paramsText.data(), paramsText.size()) &&
                    WriteFile(dir + "/mode.txt", mode, strlen(mode));
    if (!ok) {
        printf("[M25FA][FAIL] 落盘失败（%s）\n", dir.c_str());
        return 43;
    }

    // ---- 读数（非断言）：FNV-1a 64 位 + 前 4 个输出值 ----
    const size_t outUsed = (size_t)m * H_NH * H_HD * 2u;   // 只判前 m 行
    uint64_t h = 1469598103934665603ull;
    const uint8_t* ob = (const uint8_t*)hOut.data();
    for (size_t i = 0; i < outUsed; ++i) {
        h ^= ob[i];
        h *= 1099511628211ull;
    }
    printf("[M25FA] mode=%-9s m=%-5u posBase=%-5u ctx=%-5u nBlk=%-3u lane=%u blocks=%u rowsPadded=%u\n",
           mode, m, posBaseArg, ctx, a.nBlk, a.lane, blocks, rowsPadded);
    printf("[M25FA] OUT_FNV64=0x%016llx  out_bytes=%zu\n", (unsigned long long)h, outUsed);
    printf("[M25FA] out[0,0,:4]=%.6f %.6f %.6f %.6f\n", Bf16ToF(hOut[0]), Bf16ToF(hOut[1]), Bf16ToF(hOut[2]),
           Bf16ToF(hOut[3]));

    aclrtFree(dQ);
    aclrtFree(dKV);
    aclrtFree(dOut);
    aclrtFree(dP);
    aclrtFree(dWit);
    aclrtDestroyStream(stream);
    return 0;
}

// 默认档：prefill 全上下文 + decode 同档回归（可用 M25FA_M 覆盖）
inline int Main(const char* mode, const char* outDirEnv)
{
    const char* mEnv = getenv("M25FA_M");
    const char* blocksEnv = getenv("M25FA_BLOCKS");
    // 落盘根目录优先级：argv[2] > 环境变量 `M25FA_OUT` > 默认 `m25_attn_fa_core/out`
    // （README / run_checks.sh 都按 `M25FA_OUT` 工作；M189 修前该变量被忽略）。
    const char* outArg = (outDirEnv && outDirEnv[0]) ? outDirEnv : getenv("M25FA_OUT");
    const char* outEnv = (outArg && outArg[0]) ? outArg : "m25_attn_fa_core/out";

    aclInit(nullptr);
    aclrtSetDevice(0);
    int64_t aicNum = 0;
    if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicNum) != ACL_SUCCESS || aicNum <= 0) {
        aicNum = 28;
    }
    uint32_t nBlk = (uint32_t)aicNum;
    if (blocksEnv != nullptr && blocksEnv[0] != '\0') {
        const long v = strtol(blocksEnv, nullptr, 10);
        if (v > 0) { nBlk = (uint32_t)v; }
    }
    printf("[M25FA] aicNum=%d nBlk=%u mode=%s\n", (int)aicNum, nBlk, mode);

    // 默认两档：prefill m=4097（posBase=0, ctx=4097）与 decode 同档（m=1, posBase=4096, ctx=4097）
    uint32_t ms[8];
    uint32_t nCase = 0;
    if (mEnv != nullptr && mEnv[0] != '\0') {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", mEnv);
        for (char* tok = strtok(buf, ","); tok != nullptr && nCase < 8; tok = strtok(nullptr, ",")) {
            ms[nCase++] = (uint32_t)strtoul(tok, nullptr, 10);
        }
    } else {
        ms[nCase++] = 4097u;
        ms[nCase++] = 1u;
    }

    int rc = 0;
    for (uint32_t i = 0; i < nCase; ++i) {
        LaunchArgs a;
        a.m = ms[i];
        // 单请求：KV 长度 = 位置跨度；prefill 从 0 开始，m=1 档落在末尾（decode 形状）
        a.ctx = (a.m == 1u) ? 4097u : a.m;
        a.posBase = (a.m == 1u) ? (a.ctx - 1u) : 0u;
        a.lane = 0u;
        a.nBlk = nBlk;
        char params[256];
        snprintf(params, sizeof(params), "m=%u\nposBase=%u\nctx=%u\nnBlk=%u\nlane=%u\nnh=%u\nhd=%u\n", a.m,
                 a.posBase, a.ctx, a.nBlk, a.lane, H_NH, H_HD);
        char sub[256];
        snprintf(sub, sizeof(sub), "%s/m%u_%s", StripTrailingSlash(outEnv).c_str(), a.m, mode);
        const int r = RunOne("core", sub, mode, a, std::string(params));
        if (r != 0) { rc = r; }
    }
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}

}  // namespace detail
}  // namespace M15FAHost

#endif  // M15_ATTN_FA_CORE_HOST_H
