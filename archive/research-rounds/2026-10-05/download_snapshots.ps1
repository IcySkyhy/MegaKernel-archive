param([string[]]$Only)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$archive = Join-Path $workspace 'archive'
$cache = Join-Path $workspace '.git/round2-downloads'
$null = New-Item -ItemType Directory -Path $cache -Force
$plan = @(
    @('cohere-ai/cohere-megakernel','core-whole-model/cohere-megakernel'),
    @('Inferact/tpu-megakernels','core-whole-model/tpu-megakernels'),
    @('meta-pytorch/dist_moe','distributed-moe/dist_moe'),
    @('jiazhihao/mpk-apple','alternatives/mpk-apple'),
    @('WilliamZhang20/megakernel-gen','compiler-runtimes/megakernel-gen'),
    @('hoid-ai/hoid-megakernel-qwen','core-whole-model/hoid-megakernel-qwen'),
    @('thebasedcapital/latticemk','core-whole-model/latticemk'),
    @('ranvier-labs/lean-cuda-qwen','core-whole-model/lean-cuda-qwen'),
    @('kiddyboots216/training-megakernel','core-whole-model/training-megakernel'),
    @('Parth-Badgujar/transformer-megakernels','operator-scale/transformer-megakernels'),
    @('theProgrammingBox/blackwell-fp4-ffn','operator-scale/blackwell-fp4-ffn'),
    @('anishesg/persist-decode','operator-scale/persist-decode'),
    @('Beomi/husky-megakernel','long-tail-experimental/husky-megakernel'),
    @('pierre427/mlx-lm-unified','core-whole-model/mlx-lm-unified'),
    @('msaroufim/megakernels-vs-cuda-graphs','alternatives/megakernels-vs-cuda-graphs'),
    @('kamahori/vibe-megakernel','agentic-kernel-design/vibe-megakernel'),
    @('amandeepsp/megakernels','compiler-runtimes/amandeepsp-megakernels'),
    @('xys-syx/megakernel','long-tail-experimental/xys-syx-megakernel')
)
$records = @()
foreach ($item in $plan) {
    $repo = $item[0]
    if ($Only -and $repo -notin $Only) { continue }
    $destination = Join-Path $archive $item[1]
    $evidence = Get-Content -LiteralPath (Join-Path $PSScriptRoot ('candidate-evidence/' + $repo.Replace('/','--') + '.json')) -Raw | ConvertFrom-Json
    $url = 'https://codeload.github.com/' + $repo + '/zip/refs/heads/' + [uri]::EscapeDataString($evidence.branch)
    if (-not (Test-Path -LiteralPath $destination)) {
        $zip = Join-Path $cache ($repo.Replace('/','--') + '.zip')
        $stage = Join-Path $cache ($repo.Replace('/','--') + '-expanded')
        if (Test-Path -LiteralPath $stage) { throw "Staging directory already exists: $stage" }
        Invoke-WebRequest -Uri $url -OutFile $zip
        Expand-Archive -LiteralPath $zip -DestinationPath $stage
        $roots = @(Get-ChildItem -LiteralPath $stage -Directory)
        if ($roots.Count -ne 1) { throw "Unexpected archive structure for $repo" }
        $resolvedSource = $roots[0].FullName
        if (-not $resolvedSource.StartsWith($cache + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside task cache' }
        if (-not $destination.StartsWith($archive + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination outside archive' }
        Move-Item -LiteralPath $resolvedSource -Destination $destination
    }
    $files = @(Get-ChildItem -LiteralPath $destination -Recurse -Force -File)
    $record = [pscustomobject]@{
        repository=$repo; local_path=('archive/' + $item[1]); upstream=$evidence.url;
        branch=$evidence.branch; created_at=$evidence.created_at; pushed_at=$evidence.pushed_at;
        downloaded_at=(Get-Date).ToUniversalTime().ToString('o'); license=$evidence.license;
        acquisition='GitHub codeload source snapshot; no nested Git metadata; submodule bodies excluded';
        files=$files.Count; bytes=($files | Measure-Object Length -Sum).Sum
    }
    $records += $record
    $records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'downloaded-snapshots.json') -Encoding utf8
    "ARCHIVED=$repo PATH=$($item[1]) FILES=$($files.Count)"
}
