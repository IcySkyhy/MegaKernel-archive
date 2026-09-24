# Direct TK PDL ablation summary

Formal contract: H100, P32-D128, 16 layers, 80 CUDA Graph kernel nodes,
79 body edges, five warmups and 30 event-timed samples per position.  Timeline
and in-kernel timing were disabled.

| Candidate | Correct | Mean us |
| --- | --- | ---: |
| persistent VM, one kernel, three-trial mean | yes | **790.190** |
| correct 13-page/three-stage issue-trigger PDL, three-trial mean | yes | **804.409** |
| wrong 4-page/three-stage issue-trigger PDL, three-trial mean | no | 804.041 |
| wrong 4-page/three-stage arrival-trigger PDL, three-trial mean | no | 837.025 |
| correct 5-page/single-stage issue-trigger PDL, three-trial mean | yes | 841.180 |
| wrong 4-page/r48 issue-trigger PDL, three-trial mean | no | 864.555 |

The correct split PDL chain is 14.219 us / 1.799% slower than the persistent
VM body.  The 32.984-us subtraction between the two wrong four-page rows is
not an admissible trigger-timing result.  A fresh matched 13-page/three-stage
formal rerun measured 835.884 us for arrival-trigger and 801.852 us for
issue-trigger, a paired saving of **34.032 us** (0.066-us sample standard
deviation across three rotated trials).  All six processes reported 13 pages,
three input stages, 216,504 bytes of dynamic shared memory, 640 threads, no
four-page aliasing, 79 rewritten PDL edges, and passing graph-vs-eager checks
at P32/P95/P158.  Correct 13-page and wrong 4-page issue-trigger results in the
older run differed by only 0.368 us, so page aliasing and same-SM dual
admission were not the source of the speedup, but they could not establish the
trigger-timing magnitude.

Correct 13-page per-edge trial:

| PDL edge mask | Mean us | Saved vs 958.838 us completion |
| --- | ---: | ---: |
| only 1->2 | 938.321 | 20.518 |
| only 2->4 | 926.166 | 32.673 |
| only 4->5 | 936.422 | 22.417 |
| only 5->6 | 918.229 | 40.610 |
| only 6->next-1 | 916.636 | 42.203 |
| all 79 edges | 804.858 | 153.980 |

Compact-grid follow-up (fresh interleaved three-trial run):

| Physical CTAs for opcode 1/2/4/5/6 | Correct | Mean us | Delta |
| --- | --- | ---: | ---: |
| 132/132/132/132/132 | yes | **804.086** | reference |
| 132/8/128/128/132 | yes | 804.486 | +0.401 / +0.050% |

The combined compact grid is consistently slightly slower and is rejected.
The empty opcode-2 CTAs return before its eight active CTAs reach the
post-load PDL trigger, so they are not on the critical path.  Three-trial
single-op compact means are 803.918 us (opcode 2), 803.822 us (opcode 4), and
804.392 us (opcode 5); all changes are below 0.4 us versus the fresh
804.086-us reference.

## Full P32/D128 decode

The correct 13-page direct-TK body was also inserted into the real-weight
per-position decode harness.  Each of the 127 decode forwards uses its own
CUDA Graph and includes embedding, barrier reset, 80 body kernels, direct-TK
LM head, PyTorch argmax, and output-token publication.  Exactly the 79
body-to-body edges are programmatic; final down-to-LM and LM-to-argmax retain
ordinary completion dependencies.  Timeline, profiler, and device timing
records were disabled.

| Three-fresh-process mean | us / decode forward | ms / 127 forwards |
| --- | ---: | ---: |
| direct-TK 13-page PDL body | **1011.477** | **128.458** |
| original persistent megakernel | **985.121** | **125.110** |

The split candidate is 26.356 us / 2.675% slower end to end.  Individual
trial gaps were 2.659%, 2.660%, and 2.707%.

The one-forward P32 smoke produced the same token and was deterministic.
Free-running 128-token trajectories are not an exact correctness oracle for
these TK kernels: both the candidate and original persistent reference were
nondeterministic on repeated runs because their atomic reductions can change
the reduction order, and generation diverges after a low-margin argmax.
The body remains qualified by the arithmetic-preserving P32/P95/P158 tensor
checks; a teacher-forced per-position real-weight comparison is still needed
before labeling the full free-running path token-exact.

### Final opcode 6 to LM-head PDL edge

A follow-up direct-TK LM entry replaces the original global `Bar` poll with
`cudaGridDependencySynchronize()` at the activation-read dependency point.
The graph rewriter now converts exactly 80 edges: the previous 79 body edges
plus final opcode 6 to LM head. LM head to PyTorch argmax remains an ordinary
completion dependency.

The P32 one-forward smoke rewrote 80 edges, selected
`opcode7_direct_pdl_wait`, and produced the same deterministic token as the
native reference. Three fresh P32/D128 processes, with three warmups and ten
interleaved timed generations each, measured:

| Three-fresh-process mean | us / decode forward | ms / 127 forwards |
| --- | ---: | ---: |
| direct-TK body + opcode6-to-LM PDL | **1009.871** | **128.254** |
| original persistent megakernel | **984.981** | **125.093** |

The remaining gap is 24.890 us / 2.527%. Relative to the preceding
completion-edge result, candidate latency improves by 1.606 us and the
candidate-versus-native gap shrinks by 1.466 us. Therefore the final
opcode6-to-LM edge provides a small real benefit, but it does not explain the
roughly 12-us residual inferred by subtracting the separately measured
body-only scopes. That earlier attribution was too strong; the remaining
full-path residual must be isolated with matched scope controls.

### Matched-scope diagnosis of the remaining full-path gap

The previous full-path comparison launched one CUDA Graph per candidate
decode step but launched the native persistent kernel eagerly.  The
diagnostic harness now captures one native CUDA Graph per position as well
and times five matched scopes in a rotating order.  It additionally removes
LM head from both implementations to measure the 16-layer body alone.
Prefill and all cache/input preparation remain outside the timed interval.
No timeline, profiler, or in-kernel timing record was enabled.

Three fresh P32/D128 trials measured:

| Matched scope | Direct-TK PDL (us) | Persistent native (us) | Direct minus native (us) |
| --- | ---: | ---: | ---: |
| 16-layer body | **809.626** | **777.123** | **+32.503** |
| body + LM head | **976.711** | **946.086** | **+30.625** |
| complete decode step | **1009.711** | **983.127** | **+26.584** |

The complete matched gap is 26.584 us / 2.704%.  Its decomposition is:

| Increment outside the body | Direct-TK PDL (us) | Persistent native (us) | Direct minus native (us) |
| --- | ---: | ---: | ---: |
| LM-head increment | 167.085 | 168.963 | **-1.877** |
| embedding + argmax + token publication increment | 33.000 | 37.041 | **-4.041** |

Thus the entire loss is already present in the 80-node transformer body.
The opcode6-to-LM PDL edge and LM kernel are not the remaining bottleneck:
that segment is about 1.88 us faster on the split path in this matched
decomposition.  The surrounding embedding/argmax segment is also about
4.04 us faster.  These two advantages reduce the 32.503-us body deficit to
the observed 26.584-us end-to-end deficit.

Capturing the native path also removes a control mismatch.  Native
one-graph-per-step is 3.227 us faster than native eager in the same
three-trial run (983.127 versus 986.354 us).  Comparing split graphs against
native eager therefore understated the structural gap.

A matched ordinary-completion control measured the same direct-TK body at
967.162 us versus 776.942 us for persistent native.  Enabling all 79
programmatic body edges reduces the direct body to 809.626 us, recovering
about 157.536 us, or 82.8% of the original 190.220-us completion-control
deficit.  The independent five-class edge ablation recovered 158.420 us in
total, agreeing within 0.9 us despite using the position-local contract.
PDL prefetch is therefore working and accounts for almost all recoverable
cross-op overlap.

What remains is a 32.503-us body-only structural residual, averaging about
0.411 us over 79 split boundaries.  Because the direct entries preserve the
TK operator math and the cross-op counters were removed, the remaining
difference is bounded to independent-grid execution versus the persistent
CTA: repeated grid/CTA admission, per-kernel instruction and mbarrier/role
startup, loss of persistent register/SMEM state, and a grid-level dependency
gate that is coarser than the VM's intra-kernel producer/consumer protocol.
This matched experiment does not assign the 0.411-us average further among
those four startup mechanisms, but it does localize the end-to-end issue:
it is the accumulated 80-grid body structure, not LM head, argmax, embedding,
native eager launch overhead, or a missing final PDL edge.

Raw evidence is in the
[blog-v1.0 evidence release](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/releases/tag/blog-v1.0)
under `direct-tk-pdl/matched_scope_trajectory/`. The three formal PDL files
are `matched_scope_body_split_p32_d128_trial{1,2,3}.json`; the ordinary-
completion control is `matched_scope_completion_p32_d128_trial1.json`.
