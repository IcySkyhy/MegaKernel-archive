# [M129] docs 里 `.tower/` 路径引用清单（事实；交付后可达性由塔裁决）
# 命令：cd <repo root> && python3 docs/scan_quote_refs.py --tower-refs
# 说明：`.tower/**` 按主检出根解析（git worktree list 的第一项）—— 本仓 `git ls-files .tower` 的输出行数为 0。

$ python3 docs/scan_quote_refs.py --tower-refs
== docs 里 `.tower/` 路径引用清单（事实；可达性由塔裁决） ==
被扫 docs：/workspace/ascend_mega_kernel/.tower/worktrees/wt-129
`.tower/` 所在主检出：/workspace/ascend_mega_kernel
引用处数(occurrences)=19 ｜ 带路径 18 + 裸 `.tower/` 1 ｜ 去重路径(distinct)=12

| 文件:行 | `.tower/` 路径 | 本机是否存在 | git ls-files 是否跟踪 |
|---|---|---|---|
| 14-hyperconnection-ple-indexer-spec.md:491 | `.tower/comms/inbox/20260927-agent-ple-tower-review-request-m85-ple-kernel-11-11-tip-6e5d010.md` | 是 | 否 |
| 14-hyperconnection-ple-indexer-spec.md:491 | `.tower/comms/reviews/review-feat-m85-ple-kernel-spec-pin-and-standalone-i-reviewer-m85-r1.md` | 是 | 否 |
| 14-hyperconnection-ple-indexer-spec.md:729 | `.tower/worktrees/wt-25/m15_layer_loop/README.md` | 否 | 否 |
| 14-hyperconnection-ple-indexer-spec.md:730 | `.tower/comms/findings/20260926-agent-loop-improve-m13-m14-add-rmsnorm-checkpoint-2560-norm-hyper-connection.md` | 是 | 否 |
| 14-hyperconnection-ple-indexer-spec.md:813 | `.tower/comms/inbox/20260926-tower-all-crosscore-pipe-aic-mte3.md` | 是 | 否 |
| 16-vllm-ascend-qwen4exp-plan.md:707 | `.tower/worktrees/wt-37/docs/16-vllm-ascend-qwen4exp-plan.md` | 否 | 否 |
| 17-verification-standard.md:44 | `.tower/comms/inbox/20260927-agent-t3attr-tower-survey-summary-m80-layer0-t3-2-bf16-tie-kernel.md` | 是 | 否 |
| 19-gdn-prefill-scan-selection.md:801 | `.tower/worktrees/wt-104` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:806 | `.tower/worktrees/wt-104` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:809 | `.tower/worktrees/wt-104/.m18audit/m18_gdn_prefill/evidence/readme_number_audit.md` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:815 | `.tower/worktrees/wt-104` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:818 | `.tower/worktrees/wt-104/.m18audit/m18_gdn_prefill/evidence/readme_number_audit.md` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:824 | `.tower/worktrees/wt-104` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:827 | `.tower/worktrees/wt-104/.m18audit/m18_gdn_prefill/evidence/readme_number_audit.md` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:832 | `.tower/worktrees/wt-104` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:999 | `.tower/worktrees/wt-104` | 否 | 否 |
| 19-gdn-prefill-scan-selection.md:1062 | `.tower/comms/inbox/20260927-agent-prefillplan-tower-survey-summary-m103-prefill-m-4097-main-20bd20d.md` | 是 | 否 |
| 20-kernel-compliance-sweep.md:635 | `.tower/comms/findings/20260927-tower-bug-kernel-gm-setvalue-4-56-m85.md` | 是 | 否 |
| 20-kernel-compliance-sweep.md:635 | `.tower/` | 是 | 否 |

`git ls-files .tower` 输出行数 = 0
