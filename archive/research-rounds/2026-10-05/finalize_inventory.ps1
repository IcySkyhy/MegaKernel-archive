$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$items = @(Get-Content (Join-Path $PSScriptRoot 'downloaded-snapshots.json') -Raw | ConvertFrom-Json)
$additional = @(
    @('pengjh0111/TileMega','archive/compiler-runtimes/TileMega','not declared','Sparse shallow checkout; docs/experiments/** excluded; submodules not populated'),
    @('NVIDIA/TensorRT-LLM','archive/distributed-moe/TensorRT-LLM-MegaMoE','Apache-2.0','Sparse shallow checkout; CuTeDSL MegaMoE subtree and root/ancestor files only; submodules not populated'),
    @('Ascend/DeepEP','archive/distributed-moe/Ascend-DeepEP','BSD-2-Clause plus component notices','GitCode shallow checkout; submodules not populated'),
    @('liruixin_dvc/ascend_mega_kernel','archive/operator-scale/ascend_mega_kernel','not declared','GitCode shallow checkout; submodules not populated')
)
foreach ($entry in $additional) {
    $path = Join-Path $workspace $entry[1]
    $remote = git -C $path remote get-url origin
    $branch = git -C $path branch --show-current
    $date = git -C $path log -1 --format=%cI
    $items += [pscustomobject]@{
        repository=$entry[0]; local_path=$entry[1]; upstream=$remote;
        branch=$branch; created_at=$null; pushed_at=$null; checkout_commit_date=$date;
        license=$entry[2]; acquisition=$entry[3]
    }
}
$classifications = @{
    'cohere-ai/cohere-megakernel'='A-OSS'; 'Inferact/tpu-megakernels'='A-OSS';
    'meta-pytorch/dist_moe'='A-OSS (Mega subpaths; public API has peripheral launches)';
    'jiazhihao/mpk-apple'='B-OSS (multi-dispatch alternative)'; 'WilliamZhang20/megakernel-gen'='A-OSS';
    'hoid-ai/hoid-megakernel-qwen'='C-PARTIAL (binary device core)'; 'thebasedcapital/latticemk'='A-OSS';
    'ranvier-labs/lean-cuda-qwen'='C-PARTIAL (application source open, compiler backend restricted)';
    'kiddyboots216/training-megakernel'='A-OSS'; 'Parth-Badgujar/transformer-megakernels'='B-OSS';
    'theProgrammingBox/blackwell-fp4-ffn'='B-OSS (MIT plus CUTLASS BSD-3)';
    'anishesg/persist-decode'='S-SOURCE'; 'Beomi/husky-megakernel'='S-SOURCE';
    'pierre427/mlx-lm-unified'='B-OSS'; 'msaroufim/megakernels-vs-cuda-graphs'='D-BOUNDARY / subtree licenses';
    'kamahori/vibe-megakernel'='Agent-method reference / source-visible';
    'amandeepsp/megakernels'='C-PARTIAL (FX prototype, original graph execution)';
    'xys-syx/megakernel'='D-BOUNDARY / S-SOURCE (temporal fusion)';
    'pengjh0111/TileMega'='S-SOURCE'; 'NVIDIA/TensorRT-LLM'='A-OSS (MegaMoE subtree)';
    'Ascend/DeepEP'='A-OSS (experimental MegaMoE subtree)'; 'liruixin_dvc/ascend_mega_kernel'='S-SOURCE'
}
$records = foreach ($item in $items) {
    $path = Join-Path $workspace $item.local_path
    $files = @(rg --files --hidden --no-ignore -g '!**/.git/**' $path)
    $bytes = 0L
    foreach ($file in $files) { $bytes += (Get-Item -LiteralPath $file).Length }
    $window = 'creation date unavailable; see substantive date evidence in findings'
    if ($item.created_at) {
        if ([datetime]$item.created_at -ge [datetime]'2026-09-05') { $window='repository created in primary window' }
        elseif ([datetime]$item.created_at -ge [datetime]'2026-08-27') { $window='repository created in catchup window' }
        else { $window='older repository omission' }
    }
    if ($item.repository -eq 'pengjh0111/TileMega') { $window='created 2026-08-28; substantive development in primary window' }
    if ($item.repository -eq 'NVIDIA/TensorRT-LLM') { $window='existing upstream; MegaMoE update 2026-09-18; new local subtree archive' }
    $license = $item.license
    if ($item.repository -eq 'theProgrammingBox/blackwell-fp4-ffn') { $license='MIT plus CUTLASS BSD-3-Clause (root LICENSE read; API NOASSERTION)' }
    [pscustomobject]@{
        repository=$item.repository; local_path=$item.local_path; upstream=$item.upstream;
        branch=$item.branch; created_at=$item.created_at; pushed_at=$item.pushed_at;
        checkout_commit_date=$item.checkout_commit_date; temporal_class=$window;
        classification=$classifications[$item.repository]; license=$license;
        acquisition=$item.acquisition; materialized_files=$files.Count; materialized_bytes=$bytes;
        local_git_metadata_present=(Test-Path -LiteralPath (Join-Path $path '.git'))
    }
}
$document = [pscustomobject]@{
    research_date='2026-10-05'; primary_window='2026-09-05/2026-10-05';
    previous_artifacts=107; added_artifacts=$records.Count; total_artifacts=(107+$records.Count);
    generated_at=(Get-Date).ToUniversalTime().ToString('o'); artifacts=@($records)
}
$document | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $PSScriptRoot 'artifacts.json') -Encoding utf8
$search = Get-Content (Join-Path $PSScriptRoot 'github-search-results.json') -Raw | ConvertFrom-Json
$candidates = @($search.queries.items | Sort-Object full_name -Unique)
$catalog = Get-Content (Join-Path $workspace 'archive/CATALOG.md') -Raw
$screening = foreach ($candidate in $candidates) {
    $match = @($records | Where-Object repository -eq $candidate.full_name)
    $decision = 'not_selected_after_metadata_or_scope_screen; not a full source review'
    if ($match.Count) { $decision='archived_round2: ' + $match[0].classification }
    elseif ($candidate.full_name -eq 'IcySkyhy/MegaKernel-archive') { $decision='exclude_self_archive' }
    elseif ($catalog.Contains('https://github.com/' + $candidate.full_name + ')')) { $decision='already_archived_or_explicitly_discussed; see catalog/report' }
    [pscustomobject]@{repository=$candidate.full_name;url=$candidate.html_url;created_at=$candidate.created_at;decision=$decision}
}
$screening | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $PSScriptRoot 'screening-decisions.json') -Encoding utf8
"ADDED_ARTIFACTS=$($records.Count)"
"MATERIALIZED_FILES=$(($records | Measure-Object materialized_files -Sum).Sum)"
"MATERIALIZED_MIB=$([math]::Round(($records | Measure-Object materialized_bytes -Sum).Sum/1MB,2))"
"API_CANDIDATES=$($candidates.Count)"
