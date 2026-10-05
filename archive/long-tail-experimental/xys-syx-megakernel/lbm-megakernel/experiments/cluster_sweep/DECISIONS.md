# Measurement order and decisions

The supplied guide makes larger DSM implementations conditional on the launch
sweep. Stages are recorded separately and each includes an ordinary D-V2 row.
Old `results/z2` trials are context only, never pooled with this campaign.

1. **A:** Z2 C0, explicit Spread versus explicit LoadBalancing. A material
   policy improvement means at least 1% lower pooled event time at both 100
   and 1000 steps, with the same direction in every long-run process round.
   This is a practical decision threshold, not a statistical significance test.
2. **A2, conditional:** if A meets that criterion, compare existing Z2 C1/V3
   with both policies before implementing any larger DSM kernel.
3. **B:** ordinary D-V2 plus C0 Z2/XZ4/YZ6 under both policies. Query each
   actual configuration; an unsupported launch is recorded, never replaced
   by a fabricated timing. Record exact prospective producer counts separately.
4. **C/D:** choose the larger DSM experiment only after inspecting B's fixed
   cost. XZ4 is the guide's default when its C0 overhead is comparable to Z2;
   document the measured basis for the choice before writing its kernel.
5. **E, conditional:** inspect and optimize generated DSM mapping only if a
   validated clustered V3 beats D-V2. Pointer hoisting, if attempted, is a
   separate variant. If the best controlled Tt=2 cluster still loses by 2–3%,
   stop this experiment rather than adding more mechanisms.

No deeper temporal blocking is implemented as part of this sweep. It is a
separate future experiment if the stop criterion is reached.
