#!/usr/bin/env python3.12
# M164: witness that m15_prefill_epilog.h's two device regions are byte-exact extractions of the
# m28 donors. Non-zero exit on mismatch (failures propagate).
#
#   M15PE::Trans  == m28_gdn_epilog/m28_gdn_epilog.asc  lines  69..220
#   M15PE::Chain  == m28_gdn_epilog/m28_epilog_chain.asc lines  91..893
#
# Region boundaries are the marker comments in the header:
#   // >>> M15PE_TRANS_BEGIN verbatim=...   ... // <<< M15PE_TRANS_END
#   // >>> M15PE_CHAIN_BEGIN verbatim=...   ... // <<< M15PE_CHAIN_END
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def slice_lines(path, first, last):
    with open(path, encoding="utf-8") as f:
        return "".join(f.read().splitlines(keepends=True)[first - 1:last])


def region(hdr, name):
    begin = "// >>> M15PE_%s_BEGIN" % name
    end = "// <<< M15PE_%s_END" % name
    i = hdr.index(begin)
    i = hdr.index("\n", i) + 1
    j = hdr.index(end)
    return hdr[i:j]


def check(name, donor_path, first, last, hdr):
    donor = slice_lines(os.path.join(ROOT, donor_path), first, last)
    got = region(hdr, name)
    dh = hashlib.sha256(donor.encode()).hexdigest()
    gh = hashlib.sha256(got.encode()).hexdigest()
    print("%-6s donor %s:%d-%d  sha256 %s  %d bytes" % (name, donor_path, first, last, dh, len(donor)))
    print("%-6s header M15PE::%-6s sha256 %s  %d bytes" % ("", name.capitalize(), gh, len(got)))
    if got != donor:
        # show first differing line for diagnosis
        dl = donor.splitlines(keepends=True)
        gl = got.splitlines(keepends=True)
        for k in range(min(len(dl), len(gl))):
            if dl[k] != gl[k]:
                print("FAIL %s: first diff at region line %d\n  donor : %r\n  header: %r" % (name, k + 1, dl[k], gl[k]))
                break
        else:
            print("FAIL %s: length differs (donor %d lines / header %d lines)" % (name, len(dl), len(gl)))
        return False
    print("OK   %s == %s:%d-%d verbatim" % (name, donor_path, first, last))
    return True


def main():
    hdr = read_text(os.path.join(ROOT, "m15_layer_loop", "m15_prefill_epilog.h"))
    ok = True
    ok &= check("TRANS", "m28_gdn_epilog/m28_gdn_epilog.asc", 69, 220, hdr)
    ok &= check("CHAIN", "m28_gdn_epilog/m28_epilog_chain.asc", 91, 893, hdr)
    print("==== verify_lift: %s ====" % ("LIFT OK" if ok else "LIFT MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
