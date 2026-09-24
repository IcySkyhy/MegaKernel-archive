# Matched 13-page trigger-timing revalidation

The blog's previous 32.984-us value was the difference between intentionally
incorrect four-page alias controls (837.025 us arrival versus 804.041 us
issue).  It was not admissible evidence for the correct 13-page pipeline.

The replacement experiment changes only
`DIRECT_PDL_TRIGGER_WAIT_FOR_LOAD_ARRIVAL`. The two compiled entry points now
independently report their template configuration; runtime preflight verifies
that the loaded module reports 13 pages, three input-pipeline stages, 216,504
bytes of dynamic shared memory, 640 threads, and `alias_four_pages=false` for
both variants. This is a loaded-module identity check, not independent proof
of the source-level experimental design.

| Trial | 13-page arrival (us) | 13-page issue (us) | Saved (us) |
| --- | ---: | ---: | ---: |
| 1 | 836.413 | 802.310 | 34.103 |
| 2 | 835.420 | 801.399 | 34.021 |
| 3 | 835.820 | 801.848 | 33.973 |
| **Mean** | **835.884** | **801.852** | **34.032** |

The paired saving has a 0.066-us sample standard deviation and reduces the
arrival-trigger body latency by 4.071%.  Each fresh process measured positions
32--158 with five warmups and 30 event-timed samples per position.  All 18
P32/P95/P158 graph-vs-eager checks passed, all processes rewrote exactly 79
PDL edges, and all stderr files were empty.

The new 34.032-us result replaces the old 32.984-us blog value.  The mechanism
conclusion is unchanged, but its evidence now comes from a matched correct
13-page/three-stage comparison.

Raw trial JSON, stdout/stderr, preflight, and the CUDA 12.8 build log are in
the [blog-v1.0 evidence release](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/releases/tag/blog-v1.0)
under `direct-tk-pdl/13page_trigger_formal/`.
