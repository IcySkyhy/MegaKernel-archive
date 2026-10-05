// ============================================================
// m15_attn_prolog_host.h —— M88 host 侧：prolog 的**判据 + 负向对照**（新档 `runs=prolog`）
//
// 只被 m15_layer_loop.asc 在 m15_attn_kv_host.h 之后 include（用到那里的 Ctx / H_Guard / H_CmpMustDiffer）。
//
// 流程（每一步都在下面的命令表里有实跑读数）：
//   ① **权重**：按 `weights_manifest.txt` 的 `role=attn_*`（M82 落进 manifest 的 9 个 role）从
//      真实 checkpoint **pread** 进 host 平面，整块 H2D。**不做任何在线格式转换**（用户口径）。
//   ② **输入**：`evidence/attn_prolog/data/x{0,1}.bin`（host 生成的确定性 bf16，N1 输入隔离）；
//      x 缓冲按 **2 行** 分配（第 1 行全零）—— 满足 AIC 侧 Nd2Nz 的 m=1 quirk 契约
//      （见 `m15_attn_prolog_probe.h` 的 CALC_M 注释）。
//   ③ **cos/sin 表**：`data/cos_sin.bin`（host 生成，bf16）整块 H2D。**设备不算超越函数**，
//      所以旋转里**没有** `cosf/sinf` 近似这一项误差（否则 ε 里必须为它单列一项）。
//   ④ **判据**（`data/exp_*.bin` 来自 `oracle_attn_prolog.py gen`，float64 参考）：
//      · T2′（GEMM 段 y0）：逐元素 `|got−ref| ≤ 1.0·ulp(ref)`，**且**每个翻转元素必须
//        `0.5·ulp(ref) − |exact−ref| ≤ ε_acc·Σ|terms|`（ε_acc = K·2^-24 = 1.526e-4），
//        即「翻转只允许发生在真正的格点边界上」。
//      · T3（norm+rope 段 out）：`|got−ref| ≤ ε·Σ|terms| + 1.0·ulp(ref)`，ε 逐项推导见下。
//      · T1 逐字节（gate / v / raw k）：这三段是**原样抄写**（R2/R13），必须逐字节相同。
//   ⑤ **负向对照**（docs/17 §4）：AP_MODE_PLAIN_NORM / AP_MODE_ROPE_SIGN / AP_MODE_NO_ROPE 三档各跑一次，
//      **至少一条被声明有咬合力的判据必须 FAIL**；并且 x0↔x1、pos 4096↔0 的非空洞性必须有读数。
//
// 【ε 的逐项推导（docs/17 §1.1：不得拿"实测最大 X ulp"当界）】
//   norm+rope 段的每一项来源（单位 ulp = 2^-24，fp32 的 unit roundoff）：
//     norm 的两次乘（`x*rstd` 与 `*(1+w)`）              2
//     rstd：`Div` + `Sqrt` 各 ≤1 ulp（CANN RegBase 的规格档）  2
//     旋转的两个乘积（`x1·cos`、`x2·sin`）               2
//     旋转的一次加法                                     1
//     小计                                               7
//   ⇒ 取 **2× 安全系数** 覆盖上面未逐位建模的末位差 ⇒ **ε = 14·2^-24 ≈ 8.34e-7**。
//   （§1.1 的另一项 `+1.0·ulp(out)` 单列，因为参考本身已量化到 bf16 格点。）
//   注：设备侧**故意不用** NR rsqrt（`NormDonor::ComputeRstdNewtonRaphson`），改用 `Div`+`Sqrt` 的精确式，
//   这样 ε 里**没有**近似项；将来若换成 NR，必须把它的精度档重新推导进 ε。
// ============================================================

// prolog 探针的层号：任务口径「先只接 1 层（层 3）」；AttnSlot(3) == 0
constexpr uint32_t H_ApLayer() { return 3u; }


using namespace M15AP;   // host 侧：本文件的常量全部走 M15AP（与设备同一份定义）

constexpr double H_ApEpsT3 = 14.0 * (2.0 / 16777216.0);        // 14 * 2^-24 ≈ 8.34e-7
constexpr double H_ApEpsAcc = 2560.0 * (2.0 / 16777216.0);     // K * 2^-24 = 1.526e-4（GEMM 累加）

// ---- 文件读取（本档要读 host 预生成的输入/期望/表；manifest 目录是锚点，路径不随 cwd 变）----
static std::string H_ApDataDir(const std::string& manifestPath)
{
    const size_t s = manifestPath.find_last_of('/');
    const std::string dir = (s == std::string::npos) ? std::string(".") : manifestPath.substr(0, s);
    return dir + "/evidence/attn_prolog/data";
}

static bool H_ApReadFile(const std::string& path, void* dst, size_t bytes, std::string& err)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    const size_t n = fread(dst, 1, bytes, f);
    fclose(f);
    if (n != bytes) {
        err = "short read " + path;
        return false;
    }
    return true;
}

static double H_ApBf16Ulp(double v)
{
    // docs/17 §1.1：BfUlp(v) = 2^(e-7)，e = binade 指数（2^e ≤ |v| < 2^(e+1)）
    const double a = (v < 0.0) ? -v : v;
    if (a == 0.0) {
        return 0.0;
    }
    int e = 0;
    std::frexp(a, &e);          // a = m * 2^e, m ∈ [0.5,1) ⇒ binade 指数 = e-1
    return std::ldexp(1.0, (e - 1) - 7);
}

static double H_ApBf16ToF64(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return static_cast<double>(f);
}

// ---- 权重平面（host 侧组板 → H2D；**无任何数值换算**）----
static bool H_ApLoadWeights(Ctx& C, uint32_t layer, std::string& err)
{
    struct Role {
        const char* name;
        uint32_t off;
        uint32_t bytes;
    };
    const Role roles[] = {
        {"attn_q_proj", W_Q_OFF, W_Q_BYTES},           {"attn_k_proj", W_K_OFF, W_K_BYTES},
        {"attn_v_proj", W_V_OFF, W_V_BYTES},           {"attn_idx_qk_proj", W_IDX_OFF, W_IDX_BYTES},
        {"attn_q_norm", W_QN_OFF, W_QN_BYTES},         {"attn_k_norm", W_KN_OFF, W_KN_BYTES},
        {"attn_idx_q_norm", W_IQN_OFF, W_IQN_BYTES},   {"attn_idx_k_norm", W_IKN_OFF, W_IKN_BYTES},
    };
    C.apW.assign(W_BYTES, 0u);
    uint64_t total = 0;
    for (const Role& r : roles) {
        const M15Tensor* T = H_FindTensor(C.M, layer, r.name);
        if (T == nullptr) {
            printf("[m15][FAIL] manifest 无 layer=%u role=%s（M82 的 ATTN_ROLES）\n", layer, r.name);
            return false;
        }
        if (T->bytes != r.bytes || T->dtype != "BF16") {
            printf("[m15][FAIL] role=%s 尺寸/dtype 不符：manifest bytes=%llu dtype=%s，期望 %u/BF16\n",
                   r.name, static_cast<unsigned long long>(T->bytes), T->dtype.c_str(), r.bytes);
            return false;
        }
        if (!H_ReadTensor(C.M, *T, C.apWRole, err)) {
            return false;
        }
        std::memcpy(C.apW.data() + r.off, C.apWRole.data(), r.bytes);
        total += r.bytes;
    }
    H_H2D(C, C.apWDev, C.apW.data(), C.apW.size(), "ap_w");
    printf("[m15] prolog 权重：layer=%u，8 个 role %llu B（平面 %zu B，512 B 对齐）→ device\n", layer,
           static_cast<unsigned long long>(total), C.apW.size());
    return true;
}

// ---- 期望 / 输入 / 表 ----
static bool H_ApLoadData(Ctx& C, std::string& err)
{
    const std::string dir = H_ApDataDir(C.manifestPath);
    C.apExpY0[0].assign(Y0_N * 2u, 0u);
    C.apExpY0[1].assign(Y0_N * 2u, 0u);
    C.apExpOut[0].assign(AP_OUT_N * 2u, 0u);
    C.apExpOut[1].assign(AP_OUT_N * 2u, 0u);
    C.apNormOnly[0].assign(AP_OUT_N * 2u, 0u);
    C.apNormOnly[1].assign(AP_OUT_N * 2u, 0u);
    C.apExactY0[0].assign(Y0_N, 0.0f);
    C.apExactY0[1].assign(Y0_N, 0.0f);
    C.apTermsY0[0].assign(Y0_N, 0.0f);
    C.apTermsY0[1].assign(Y0_N, 0.0f);
    C.apTermsOut[0].assign(AP_OUT_N, 0.0f);
    C.apTermsOut[1].assign(AP_OUT_N, 0.0f);
    C.apIn.assign(2u * AP_HIDDEN * 2u, 0u);                  // 2 行：行 0 = x0、行 1 = x1（第 2 行是 filler）
    C.apCs.assign(AP_CS_BYTES, 0u);
    C.apY0.assign(Y0_N * 2u, 0u);
    C.apOut.assign(AP_OUT_N * 2u, 0u);

    for (uint32_t i = 0; i < 2u; ++i) {
        const std::string suf = (i == 0u) ? "x0" : "x1";
        if (!H_ApReadFile(dir + "/" + suf + ".bin", C.apIn.data() + i * AP_HIDDEN * 2u, AP_HIDDEN * 2u, err) ||
            !H_ApReadFile(dir + "/exp_y0_" + suf + ".bin", C.apExpY0[i].data(), Y0_N * 2u, err) ||
            !H_ApReadFile(dir + "/exp_out_" + suf + ".bin", C.apExpOut[i].data(), AP_OUT_N * 2u, err) ||
            !H_ApReadFile(dir + "/exp_normonly_" + suf + ".bin", C.apNormOnly[i].data(), AP_OUT_N * 2u, err) ||
            !H_ApReadFile(dir + "/exp_y0_exact_" + suf + ".bin", C.apExactY0[i].data(), Y0_N * 4u, err) ||
            !H_ApReadFile(dir + "/exp_terms_y0_" + suf + ".bin", C.apTermsY0[i].data(), Y0_N * 4u, err) ||
            !H_ApReadFile(dir + "/exp_terms_out_" + suf + ".bin", C.apTermsOut[i].data(), AP_OUT_N * 4u, err)) {
            printf("[m15][FAIL] 读取期望失败：%s（先在 evidence/attn_prolog 跑 "
                   "`python3.12 oracle_attn_prolog.py gen`）\n", err.c_str());
            return false;
        }
    }
    if (!H_ApReadFile(dir + "/cos_sin.bin", C.apCs.data(), AP_CS_BYTES, err)) {
        printf("[m15][FAIL] 读取 cos/sin 表失败：%s\n", err.c_str());
        return false;
    }
    H_H2D(C, C.apXDev, C.apIn.data(), C.apIn.size(), "ap_x");
    H_H2D(C, C.apCsDev, C.apCs.data(), C.apCs.size(), "ap_cs");
    printf("[m15] prolog 输入/表：x 2 行 × %u bf16、cos/sin 表 %u 行 × %u bf16（%zu B）→ device\n", AP_HIDDEN,
           CS_NPOS, CS_ROW_ELEMS, C.apCs.size());
    return true;
}

// ---- 非空洞性（docs/17 §4）：比对面必须非零非退化，**判据的比对面一律取自落盘/读回的字节** ----
static bool H_ApNonVacuity(Ctx& C, const std::vector<uint8_t>& expOut, const std::vector<uint8_t>& expY0,
                           const char* who)
{
    uint32_t nzOut = 0;
    uint32_t nzY0 = 0;
    for (size_t i = 0; i < expOut.size(); i += 2u) {
        if (expOut[i] != 0u || expOut[i + 1u] != 0u) {
            ++nzOut;
        }
    }
    for (size_t i = 0; i < expY0.size(); i += 2u) {
        if (expY0[i] != 0u || expY0[i + 1u] != 0u) {
            ++nzY0;
        }
    }
    bool ok = true;
    ok = H_Guard(C, nzOut > AP_OUT_N / 2u) && ok;
    ok = H_Guard(C, nzY0 > Y0_N / 2u) && ok;
    printf("[m15]   Ap.nonvac.%s：期望非零 out %u/%u、y0 %u/%u（整片为 0 会让判据空过 ⇒ 这是 guard）\n",
           who, nzOut, AP_OUT_N, nzY0, Y0_N);
    return ok;
}

// ---- 单段比较（返回 FAIL 条数；counted=true 时计入 C.checks/C.fails 并打印 PASS/FAIL）----
// kind: 0 = T1 逐字节；1 = T3；2 = T2′（GEMM）
static uint32_t H_ApCmpSeg(Ctx& C, const char* tag, const uint8_t* got, const uint8_t* exp, uint32_t elems,
                           const float* terms, const float* exact, uint32_t kind, bool counted, bool verbose)
{
    uint32_t nFail = 0;
    uint32_t nDiff = 0;
    uint32_t firstBad = 0xFFFFFFFFu;
    double worstAbs = 0.0;
    double worstSlack = 0.0;

    for (uint32_t i = 0; i < elems; ++i) {
        uint16_t gb;
        uint16_t eb;
        std::memcpy(&gb, got + 2u * i, 2);
        std::memcpy(&eb, exp + 2u * i, 2);
        const double g = H_ApBf16ToF64(gb);
        const double e = H_ApBf16ToF64(eb);
        if (gb != eb) {
            ++nDiff;
        }
        bool bad = false;
        if (kind == 0u) {
            bad = (gb != eb);
        } else {
            const double ulp = H_ApBf16Ulp(e);
            const double absd = (g > e) ? (g - e) : (e - g);
            if (kind == 1u) {
                // T3：|got−ref| ≤ ε·Σ|terms| + 1.0·ulp(ref)
                const double bound = H_ApEpsT3 * static_cast<double>(terms[i]) + 1.0 * ulp;
                bad = (absd > bound);
                if (absd - bound > worstAbs) {
                    worstAbs = absd - bound;
                }
            } else {
                // T2′：① 逐元素 ≤ 1.0·ulp；② 翻转元素必须贴格点中点（在累加误差之内）
                if (absd > 1.0 * ulp) {
                    bad = true;
                    if (absd - 1.0 * ulp > worstAbs) {
                        worstAbs = absd - 1.0 * ulp;
                    }
                } else if (gb != eb) {
                    const double ex = static_cast<double>(exact[i]);
                    const double dmid = 0.5 * ulp - ((ex > e) ? (ex - e) : (e - ex));
                    const double slack = dmid - H_ApEpsAcc * static_cast<double>(terms[i]);
                    if (slack > 0.0) {
                        bad = true;   // 翻转得离中点太远 ⇒ 不是累加误差能解释的
                        if (slack > worstSlack) {
                            worstSlack = slack;
                        }
                    }
                }
            }
        }
        if (bad) {
            ++nFail;
            if (firstBad == 0xFFFFFFFFu) {
                firstBad = i;
            }
        }
    }

    if (counted) {
        C.checks++;
        if (nFail != 0u) {
            C.fails++;
        }
    }
    if (verbose && nFail != 0u && kind == 1u && elems % 256u == 0u && elems >= 24u * 256u) {
        // 报告项：按 256 维头逐头给「不同元素数」，用来区分「全头都错」与「只有某些头错」
        printf("[m15]   %-28s 逐头不同元素数：", tag);
        for (uint32_t hh = 0; hh * 256u < elems && hh < 24u; ++hh) {
            uint32_t d = 0;
            for (uint32_t j = 0; j < 256u; ++j) {
                if (std::memcmp(got + (hh * 256u + j) * 2u, exp + (hh * 256u + j) * 2u, 2) != 0) {
                    ++d;
                }
            }
            printf(" h%u=%u", hh, d);
        }
        printf("\n");
    }
    if (verbose) {
        const char* kn = (kind == 0u) ? "T1字节" : ((kind == 1u) ? "T3" : "T2'");
        printf("[m15]   %-28s %-6s %s  n=%u 差异=%u 越界=%u", tag, kn, (nFail == 0u) ? "PASS" : "FAIL", elems,
               nDiff, nFail);
        if (nFail != 0u) {
            const uint32_t j = (firstBad == 0xFFFFFFFFu) ? 0u : firstBad;
            uint16_t gb;
            uint16_t eb;
            std::memcpy(&gb, got + 2u * j, 2);
            std::memcpy(&eb, exp + 2u * j, 2);
            printf("  首错 #%u got=%.6g ref=%.6g Σ|terms|=%.6g", j, H_ApBf16ToF64(gb), H_ApBf16ToF64(eb),
                   (terms == nullptr) ? 0.0 : static_cast<double>(terms[j]));
        }
        printf("\n");
    }
    (void)worstAbs;
    (void)worstSlack;
    return nFail;
}

// 全段比较（got 是 device 读回的 y0/out 两块平面）
static uint32_t H_ApCmpAll(Ctx& C, const uint8_t* gotY0, const uint8_t* gotOut, uint32_t idxIn, bool counted,
                           bool verbose, bool cmpOut = true)
{
    uint32_t f = 0;
    const uint8_t* eY0 = C.apExpY0[idxIn].data();
    const uint8_t* eOut = C.apExpOut[idxIn].data();
    const float* tY0 = C.apTermsY0[idxIn].data();
    const float* tOut = C.apTermsOut[idxIn].data();
    const float* exY0 = C.apExactY0[idxIn].data();

    // y0（GEMM 的 4 段）—— T2′
    f += H_ApCmpSeg(C, "Ap.y0.qg", gotY0 + Y0_QG * 2u, eY0 + Y0_QG * 2u, QG_W, tY0 + Y0_QG, exY0 + Y0_QG, 2u,
                    counted, verbose);
    f += H_ApCmpSeg(C, "Ap.y0.k", gotY0 + Y0_K * 2u, eY0 + Y0_K * 2u, NKV * HD, tY0 + Y0_K, exY0 + Y0_K, 2u,
                    counted, verbose);
    f += H_ApCmpSeg(C, "Ap.y0.v", gotY0 + Y0_V * 2u, eY0 + Y0_V * 2u, NKV * HD, tY0 + Y0_V, exY0 + Y0_V, 2u,
                    counted, verbose);
    f += H_ApCmpSeg(C, "Ap.y0.idx", gotY0 + Y0_IDX * 2u, eY0 + Y0_IDX * 2u, IDX_W, tY0 + Y0_IDX, exY0 + Y0_IDX,
                    2u, counted, verbose);

    // out（prolog 的 6 段）
    if (!cmpOut) {
        return f;   // pos 档：out 段与 pos 有关，期望是对另一个 pos 算的，不能比
    }
    f += H_ApCmpSeg(C, "Ap.out.q", gotOut + OUT_Q * 2u, eOut + OUT_Q * 2u, QW, tOut + OUT_Q, nullptr, 1u,
                    counted, verbose);
    // ⚠ **必须是 `AP_OUT_K`，不能裸写 `OUT_K`**：本 TU 有 `using namespace M15G;`，裸写会静默取
    //   `M15G::OUT_K` = 6144 ⇒ 判据会去比 **gate 段**，k 段从未被比较（M88 r2 复审抓到的假绿）。
    //   反假绿审计见 `evidence/attn_prolog/audit_bare_names.py`（它现在 0 命中；修前 3 命中）。
    f += H_ApCmpSeg(C, "Ap.out.k", gotOut + AP_OUT_K * 2u, eOut + AP_OUT_K * 2u, NKV * HD, tOut + AP_OUT_K,
                    nullptr, 1u, counted, verbose);
    f += H_ApCmpSeg(C, "Ap.out.qidx", gotOut + OUT_QIDX * 2u, eOut + OUT_QIDX * 2u, IDX_Q, tOut + OUT_QIDX,
                    nullptr, 1u, counted, verbose);

    // 三段**原样抄写**（R2/R13）。判据分两层，**两层都要**：
    //   ① **T1 逐字节 vs 设备自己的 y0 段** ⇒ 证明这条通道真的没被 norm/rope 碰过
    //      （gate 在 y0 里按头交织：源 = Y0_QG + h*2*HD + HD，宽 256；dst 是紧凑的 h*256）
    //   ② **T2′ vs oracle**（沿用 y0 段的 exact/terms，误差模型与 GEMM 同）⇒ 证明抄过去的**值**也对
    // 只做 ① 会漏「AIC 与 AIV 一起错」；只做 ② 会漏「AIV 多碰了一下」。
    {
        uint32_t gateBad = 0;
        for (uint32_t hh = 0; hh < NH; ++hh) {
            if (std::memcmp(gotOut + (OUT_GATE + hh * HD) * 2u, gotY0 + (Y0_QG + hh * 2u * HD + HD) * 2u,
                            HD * 2u) != 0) {
                ++gateBad;
            }
        }
        const bool okv = (std::memcmp(gotOut + OUT_V * 2u, gotY0 + Y0_V * 2u, NKV * HD * 2u) == 0);
        const bool okr = (std::memcmp(gotOut + OUT_KRAW * 2u, gotY0 + (Y0_IDX + IDX_Q) * 2u, IDX_D * 2u) == 0);
        if (counted) {
            C.checks += 3u;
            if (gateBad != 0u) {
                C.fails++;
            }
            if (!okv) {
                C.fails++;
            }
            if (!okr) {
                C.fails++;
            }
        }
        f += (gateBad != 0u ? 1u : 0u) + (okv ? 0u : 1u) + (okr ? 0u : 1u);
        if (verbose) {
            printf("[m15]   Ap.copy.gate                  T1字节 %s  n=%u（24 个头 × 256，vs 设备自己的 y0 交织源）坏头=%u\n",
                   (gateBad == 0u) ? "PASS" : "FAIL", QW, gateBad);
            printf("[m15]   Ap.copy.v                     T1字节 %s  n=%u（vs 设备自己的 y0 段）\n",
                   okv ? "PASS" : "FAIL", NKV * HD);
            printf("[m15]   Ap.copy.kraw                  T1字节 %s  n=%u（vs 设备自己的 y0 段）\n",
                   okr ? "PASS" : "FAIL", IDX_D);
        }
    }
    // ⚠ 这三段**不能**用「T1 逐字节 vs oracle」：它们是 y0 的原样抄写（由 `Ap.copy.*` 用
    //   T1 逐字节 vs **设备自己的 y0** 证明），而设备 y0 自己就可能与 oracle 差 1 ulp（`Ap.y0.qg`
    //   已报 2 个真格点边界元素）。拿 oracle 的字节去判它们，会把「GEMM 的 1-ulp 边界差」误判成
    //   「抄写错」。⇒ 值与 GEMM 同模型：**T2′ vs oracle**，`terms`/`exact` 取自对应的 y0 区段。
    //   gate 在 y0 里按头交织（源 = Y0_QG + h*2*HD + HD，宽 HD），所以要**压紧**成 24×HD 的向量。
    {
        static std::vector<float> tGate;
        static std::vector<float> eGate;
        tGate.resize(QW);
        eGate.resize(QW);
        for (uint32_t hh = 0; hh < NH; ++hh) {
            for (uint32_t j = 0; j < HD; ++j) {
                tGate[hh * HD + j] = tY0[Y0_QG + hh * 2u * HD + HD + j];
                eGate[hh * HD + j] = exY0[Y0_QG + hh * 2u * HD + HD + j];
            }
        }
        f += H_ApCmpSeg(C, "Ap.out.gate", gotOut + OUT_GATE * 2u, eOut + OUT_GATE * 2u, QW, tGate.data(),
                        eGate.data(), 2u, counted, verbose);
    }
    f += H_ApCmpSeg(C, "Ap.out.v", gotOut + OUT_V * 2u, eOut + OUT_V * 2u, NKV * HD, tY0 + Y0_V, exY0 + Y0_V, 2u,
                    counted, verbose);
    f += H_ApCmpSeg(C, "Ap.out.kraw", gotOut + OUT_KRAW * 2u, eOut + OUT_KRAW * 2u, IDX_D, tY0 + Y0_IDX + IDX_Q,
                    exY0 + Y0_IDX + IDX_Q, 2u, counted, verbose);
    return f;
}

// 报告用：两段 bf16 的不同元素数（不计数、不断言）
static uint32_t H_ApCountDiff(const uint8_t* a, const uint8_t* b, uint32_t elems)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < elems; ++i) {
        if (std::memcmp(a + 2u * i, b + 2u * i, 2) != 0) {
            ++n;
        }
    }
    return n;
}

static uint16_t rd16(const uint8_t* p)
{
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

static void H_ApLaunch(Ctx& C, uint32_t xRow, uint32_t pos, uint32_t mode, uint32_t dbg = 0u)
{
    m15_attn_prolog_probe_kernel<<<C.numBlocks, 0, C.stream>>>(
        reinterpret_cast<uint8_t*>(C.apWDev), reinterpret_cast<uint8_t*>(C.apXDev) + xRow * AP_HIDDEN * 2u,
        reinterpret_cast<uint8_t*>(C.apCsDev), reinterpret_cast<uint8_t*>(C.apY0Dev),
        reinterpret_cast<uint8_t*>(C.apOutDev), pos, mode, dbg);
}

// **每次 launch 前把两块读回平面毒成 0xCD**。
// 为什么必须做：这两块 GM 是跨 launch 复用的，若某次 launch「什么都没写」，留在里面的是**上一次
// 的正确结果**，判据会**照常 PASS**（= 假绿）。毒化之后「没写」会读回 0xCD 而不是正确值，
// 判据才真的在比「本次 launch 产出的字节」。这也是变异对照（NO_COPY / NO_GEMM）能咬住的前提。
static void H_ApPoisonOut(Ctx& C)
{
    const size_t y0b = static_cast<size_t>(Y0_N) * 2u;
    const size_t ob = static_cast<size_t>(AP_OUT_N) * 2u;
    aclrtMemset(C.apY0Dev, y0b, 0xCD, y0b);
    aclrtMemset(C.apOutDev, ob, 0xCD, ob);
}

static bool H_ApRunOnce(Ctx& C, uint32_t xRow, uint32_t pos, uint32_t mode, uint32_t* failsOut,
                        uint32_t dbg = 0u, bool cmpOut = true)
{
    H_ApPoisonOut(C);
    H_ApLaunch(C, xRow, pos, mode, dbg);
    if (!H_Sync(C, "ap_sync")) {
        return false;
    }
    H_D2H(C, C.apY0.data(), C.apY0Dev, C.apY0.size(), "ap_y0");
    H_D2H(C, C.apOut.data(), C.apOutDev, C.apOut.size(), "ap_out");
    // cmpOut=false：本档的 pos 与期望的 pos 不同 ⇒ 只比与 pos 无关的 y0 段，跳过 out 段
    const uint32_t f = H_ApCmpAll(C, C.apY0.data(), C.apOut.data(), xRow,
                                  mode == AP_MODE_CONTRACT && cmpOut,
                                  mode == AP_MODE_CONTRACT && cmpOut, cmpOut);
    if (failsOut != nullptr) {
        *failsOut = f;
    }
    return true;
}

// ============================================================
// 验证 prolog 的总入口
// ============================================================
static bool H_RunProlog(Ctx& C)
{
    printf("\n[m15] ===== 验证 prolog：attention 前端（主干 q/k + indexer）的数值语义（M88）=====\n");
    printf("[m15]   ⚠⚠ 本档**当前跑不通**（rc=124 挂死）：AIC 的 mode-2 CrossCoreSetFlag 与 AIV 侧的\n");
    printf("[m15]        CrossCoreWaitFlag 在本 kernel 里**没能配对上**（AIC 单方面 set 也会阻塞）。\n");
    printf("[m15]        本档**不在 runs=all 里**，必须显式 runs=prolog；runs=all 仍 2068 判定项 0 FAIL。\n");
    printf("[m15]        逐档诊断读数见 evidence/attn_prolog_device_bisect.log；能跑通的部分（host 参考）见\n");
    printf("[m15]        evidence/attn_prolog_oracle_selfcheck.log。\n");
    // ⚠ **M97（2026-09-27）对下面 5 行的时点更正 —— 原文一字未动，更正紧随其后**。
    //   上面那段「本档当前跑不通（rc=124 挂死）… 根因在 AIC/AIV 的 mode-2 配对」记的是 M88
    //   当时（`e36e1d1`）的事实；它**在同一个 commit 之后的 r2 就被解掉了**，而且**真根因不是
    //   mode-2 配对**，是本文件自己的一个花括号 —— AIV 那条臂曾被整块写在 `if ASCEND_IS_AIC {`
    //   内部 ⇒ AIV 核上整段被编译掉、AIC 却会等一个只有 AIV 能置的 flag ⇒ 死锁。
    //   修后本档 rc=0 / ALL PASS（逐条见 `evidence/attn_prolog_r4_runs.log` 与 README §4.0/§4.2）。
    //   之所以不直接删掉原文：本仓的惯例是「保留原文 + 标注时点的更正」，读者才能看到
    //   「M88 当时以为卡在哪」；**但请不要据此以为本档现在跑不通**。
    printf("[m15]   ⚠ 上面那 2 行（rc=124 / mode-2 配对）是 M88 当时的原话，**已过期**：挂死在\n");
    printf("[m15]      同一个 commit 之后的 r2 就修好了，真根因是本文件的一个花括号（AIV 臂被整块写在\n");
    printf("[m15]      `if ASCEND_IS_AIC {` 内部），与 mode-2 配对**无关**；修后本档 rc=0 / ALL PASS\n");
    printf("[m15]      （见 evidence/attn_prolog_r4_runs.log 与 README §4.0/§4.2）。\n");
    printf("[m15]      ⇒ 下面同一次运行打出的判据才是本档的当前读数。上面第 3-5 行（不在 runs=all 里 /\n");
    printf("[m15]        诊断读数位置）仍然成立。\n");
    printf("[m15]   规则来源（N2 外部权威，逐条 file:symbol 见 m15_attn_prolog.h 的 R1-R13）：\n");
    printf("[m15]     R1/R2 qkv split（q vs gate **按头交织**）  qwen3_next.py:427-434、qsa.py:505-506\n");
    printf("[m15]     R3    q/k norm = **GemmaRMSNorm**（乘 1+w，不是乘 w） qsa.py:350-351、layernorm.py:140-168\n");
    printf("[m15]     R4/R5 文本-only ⇒ 三个 MRoPE 轴相同 ⇒ 退化为普通 NeoX partial RoPE\n");
    printf("[m15]           model.py:846-852、mrope.py:236-247\n");
    printf("[m15]     R7-R9 rotary_dim=64、cos/sin 表 bf16、NeoX 配对 (j,j+32)\n");
    printf("[m15]           rotary_embedding/__init__.py:68-71、base.py:80-125、common.py:134-173\n");
    printf("[m15]     R10-R13 indexer 640 宽、GemmaRMSNorm(128)、同一张表只旋前 64 维、raw k 原样\n");
    printf("[m15]           indexer_qsa.py:131-145、qsa_pre_indexer.py:31-32,69,72-79,367-372\n");
    printf("[m15]   **基准交代**：本段判的是**前端算子语义**；本工程不做打分/topk/expand，"
           "**任何地方不得写「m=4097 对齐官方输出」** —— 官方 QSA 每 token 只 attend 约一半历史"
           "（docs/17:281）。\n");
    const uint32_t checkBase = C.checks;
    const uint32_t guardBase = C.guards;
    bool ok = true;
    std::string err;
    // 诊断闸：M15_AP_DBG 直接作为 kernel 的 dbg 参数**跑完整的判据**（不再「只跑一次就返回」——
    // 那个形态会让每个 dbg 档都拿不到判据读数）。语义见 m15_attn_prolog_probe.h。
    uint32_t apDbg = 0u;
    if (const char* e = getenv("M15_AP_DBG")) {
        apDbg = static_cast<uint32_t>(strtoul(e, nullptr, 10));
        printf("[m15]   [DBG] 本次全部 launch 用 dbg=%u（诊断档，判据照跑）\n", apDbg);
    }

    ok = H_ApLoadWeights(C, H_ApLayer(), err) && ok;
    ok = H_ApLoadData(C, err) && ok;
    if (!ok) {
        printf("[m15] 验证 prolog：前置（权重/输入/期望）失败，本段判定项 %u 条\n", C.checks - checkBase);
        return false;
    }

    // ---- 非空洞性（比对面取自**读回的期望文件**；空过是比没有判据更糟的形态）----
    ok = H_ApNonVacuity(C, C.apExpOut[0], C.apExpY0[0], "x0") && ok;
    ok = H_ApNonVacuity(C, C.apExpOut[1], C.apExpY0[1], "x1") && ok;

    // ---- 契约档（mode 0）：x0 / x1 各一次 ----
    uint32_t f0 = 0;
    uint32_t f1 = 0;
    if (!H_ApRunOnce(C, 0u, 4096u, AP_MODE_CONTRACT, &f0, apDbg)) {
        return false;
    }
    // ⚠ 必须在**第一次跑完**就把读回值拷出来：此前 `outX0`/`outX1` 都在两次跑之后才取 ⇒ 两者恒等
    //   ⇒ `Ap.nonvac.sens` 变成一条**恒 FAIL 的假判据**（本 mission r2 修）
    std::vector<uint8_t> outX0 = C.apOut;
    if (!H_ApRunOnce(C, 1u, 4096u, AP_MODE_CONTRACT, &f1, apDbg)) {
        return false;
    }
    std::vector<uint8_t> outX1 = C.apOut;
    ok = H_Guard(C, f0 == 0u && f1 == 0u) && ok;

    // ---- 同配置重跑（报告项）：区分「首次 launch 才坏」与「真不确定」----
    uint32_t fRep = 0;
    if (!H_ApRunOnce(C, 0u, 4096u, AP_MODE_CONTRACT, &fRep, apDbg)) {
        return false;
    }
    ok = H_Guard(C, fRep == 0u) && ok;
    const uint32_t repDiff = H_ApCountDiff(C.apOut.data() + OUT_Q * 2u, outX0.data() + OUT_Q * 2u, QW);
    printf("[m15]   Ap.repeat                   报告项  同配置重跑与首次的 q 段不同元素 = %u/%u\n", repDiff, QW);

    // ---- 非空洞性：改输入必须改输出 ----
    ok = H_CmpMustDiffer(C, "Ap.nonvac.sens", outX0.data() + OUT_Q * 2u, outX1.data() + OUT_Q * 2u, QW * 2u) && ok;

    // ---- 非空洞性：改位置必须改输出（证明旋转真的用了 pos）----
    uint32_t fPos0 = 0;
    if (!H_ApRunOnce(C, 0u, 0u, AP_MODE_CONTRACT, &fPos0, apDbg, /*cmpOut=*/false)) {
        return false;
    }
    ok = H_Guard(C, fPos0 == 0u) && ok;   // pos 0 也是契约档（只比与 pos 无关的 y0 段）
    ok = H_CmpMustDiffer(C, "Ap.nonvac.pos", outX0.data() + OUT_Q * 2u, C.apOut.data() + OUT_Q * 2u, QW * 2u) && ok;

    // ---- 负向对照（docs/17 §4）：三档**方向级**改错，每档必须至少一条判据 FAIL ----
    const uint32_t modes[] = {AP_MODE_PLAIN_NORM, AP_MODE_ROPE_SIGN, AP_MODE_NO_ROPE};
    const char* modeName[] = {"PLAIN_NORM", "ROPE_SIGN", "NO_ROPE"};
    uint32_t negOk = 0;
    for (uint32_t m = 0; m < 3u; ++m) {
        uint32_t fNeg = 0;
        if (!H_ApRunOnce(C, 0u, 4096u, modes[m], &fNeg)) {
            return false;
        }
        printf("[m15]   负向对照 mode=%u(%s)：被打破的判据 %u/%u 条\n", modes[m], modeName[m], fNeg, 10u);
        if (fNeg > 0u) {
            ++negOk;
        }
    }
    ok = H_Guard(C, negOk == 3u) && ok;   // 三档都必须被打破；否则判据对该方向没有判别力

    // ---- 变异对照（塔的纪律：**把被测对象弄坏，判据必须变红**）----
    // 与上面「负向对照」不同：负向对照改的是**算件的规则方向**（norm 的 (1+w)、rope 的符号），
    // 这里改的是**被判对象本身的存在性**——直接让 AIV 不写抄写段、让 AIC 不跑 GEMM。
    // 判据若在这两档下还是绿的，就说明它比的根本不是那个区段（r2 的 `OUT_K` 假绿正是这样）。
    {
        const uint32_t muts[] = {AP_MODE_NO_COPY, AP_MODE_NO_GEMM};
        const char* mutName[] = {"NO_COPY(抄写段不写)", "NO_GEMM(AIC 不跑 GEMM)"};
        for (uint32_t m = 0; m < 2u; ++m) {
            uint32_t fMut = 0;
            if (!H_ApRunOnce(C, 0u, 4096u, muts[m], &fMut)) {
                return false;
            }
            printf("[m15]   变异对照 mode=%u(%s)：被打破的判据 %u 条\n", muts[m], mutName[m], fMut);
            ok = H_Guard(C, fMut > 0u) && ok;   // 必须变红
        }
    }

    // ---- 诊断（**报告项，不计入判定**）：把 device 的 mode=3（NO_ROPE）读数与 oracle 的
    //      「norm 后 / rope 前」对比 ⇒ 直接区分「norm 错」还是「rope 错」。
    uint32_t fDiag = 0;
    if (H_ApRunOnce(C, 0u, 4096u, AP_MODE_NO_ROPE, &fDiag)) {
        const uint8_t* nrm = C.apNormOnly[0].data();
        printf("[m15]   [诊断] device(mode=3 NO_ROPE) vs oracle(仅 norm)："
               "q 不同 %u/%u、k 不同 %u/%u、qidx 不同 %u/%u\n",
               H_ApCountDiff(C.apOut.data() + OUT_Q * 2u, nrm + OUT_Q * 2u, QW), QW,
               H_ApCountDiff(C.apOut.data() + AP_OUT_K * 2u, nrm + AP_OUT_K * 2u, NKV * HD), NKV * HD,
               H_ApCountDiff(C.apOut.data() + OUT_QIDX * 2u, nrm + OUT_QIDX * 2u, IDX_Q), IDX_Q);
        // 数值级诊断：q 头 0 的前 8 个元素（设备 mode=3 / oracle-norm / oracle-out）
        std::vector<uint8_t> outRope = outX0;   // 第一次契约跑的 mode=0 读数
        printf("[m15]   [诊断] q 头0：dev(mode0) / dev(mode3) / oracle-norm / oracle-out(rope) / oracle-norm[+32]\n");
        for (uint32_t j = 0; j < 8u; ++j) {
            printf("[m15]     #%u  %.6g  %.6g  %.6g  %.6g  %.6g\n", j,
                   H_ApBf16ToF64(rd16(outRope.data() + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(C.apOut.data() + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(nrm + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(C.apExpOut[0].data() + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(nrm + (OUT_Q + 32u + j) * 2u)));
        }
        printf("[m15]   [诊断] q 头0 前 8 元素：\n");
        for (uint32_t j = 0; j < 8u; ++j) {
            printf("[m15]     #%u  dev(mode3)=%.6g   oracle-norm=%.6g   oracle-out(rope)=%.6g\n", j,
                   H_ApBf16ToF64(rd16(C.apOut.data() + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(nrm + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(C.apExpOut[0].data() + (OUT_Q + j) * 2u)));
        }
        printf("[m15]   [诊断] q 头0 元素 32..35（rope 的另一半）：\n");
        for (uint32_t j = 32u; j < 36u; ++j) {
            printf("[m15]     #%u  dev(mode3)=%.6g   oracle-norm=%.6g   oracle-out(rope)=%.6g\n", j,
                   H_ApBf16ToF64(rd16(C.apOut.data() + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(nrm + (OUT_Q + j) * 2u)),
                   H_ApBf16ToF64(rd16(C.apExpOut[0].data() + (OUT_Q + j) * 2u)));
        }
    }

    printf("[m15]   覆盖交代：本段**不覆盖** ① 打分/topk/expand（packed 的选择语义）；② attention 核心；"
           "③ `o_proj`；④ 把 prolog 接进 `M15L_FusedBody` 的接口（需改 scope 外的 m15_hc_host.h）。\n");
    C.phaseProlog = C.checks - checkBase;
    printf("[m15] 验证 prolog：本段判定项 %u 条 + guard %u 条（累计 checks %u / guards %u / fails %u）\n",
           C.phaseProlog, C.guards - guardBase, C.checks, C.guards, C.fails);
    return ok;
}

