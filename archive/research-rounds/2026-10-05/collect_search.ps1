param([switch]$Refresh)
$ErrorActionPreference = 'Stop'
$outputDir = $PSScriptRoot
$headers = @{ Accept = 'application/vnd.github+json'; 'User-Agent' = 'MegaKernel-research-round2' }
$credentialLines = "protocol=https`nhost=github.com`n`n" | git credential fill
$credential = @{}
foreach ($line in $credentialLines) {
    $split = $line.IndexOf('=')
    if ($split -gt 0) { $credential[$line.Substring(0, $split)] = $line.Substring($split + 1) }
}
if ($credential.ContainsKey('password')) { $headers.Authorization = "Bearer $($credential['password'])" }
$queries = @(
    'megakernel created:2026-09-05..2026-10-05',
    'megakernel created:2026-08-27..2026-09-04',
    'megakernel pushed:2026-09-05..2026-10-05',
    '"mega-kernel" created:2026-08-27..2026-10-05',
    '"persistent kernel" created:2026-08-27..2026-10-05',
    '"megakernel" in:readme created:2026-09-05..2026-10-05',
    '"persistent" "MoE" created:2026-08-27..2026-10-05',
    '"fused" "MoE" created:2026-09-05..2026-10-05',
    '"whole-model" created:2026-09-05..2026-10-05'
)
$results = @()
foreach ($q in $queries) {
    try {
        $uri = 'https://api.github.com/search/repositories?q=' + [uri]::EscapeDataString($q) + '&sort=updated&per_page=100'
        $response = Invoke-RestMethod -Uri $uri -Headers $headers
        $items = @($response.items | Select-Object full_name,html_url,description,created_at,pushed_at,updated_at,size,stargazers_count,default_branch,fork,archived,@{n='license';e={$_.license.spdx_id}})
        $results += [pscustomobject]@{ query=$q; total_count=$response.total_count; incomplete_results=$response.incomplete_results; retrieved=$items.Count; items=$items }
        Write-Output "QUERY=$q TOTAL=$($response.total_count) RETRIEVED=$($items.Count)"
    } catch {
        $results += [pscustomobject]@{ query=$q; error=$_.Exception.Message }
        Write-Output "QUERY_ERROR=$q $($_.Exception.Message)"
    }
}
$document = [pscustomobject]@{ retrieved_at=(Get-Date).ToUniversalTime().ToString('o'); primary_window='2026-09-05/2026-10-05'; catchup_window='2026-08-27/2026-09-04'; queries=$results }
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputDir 'github-search-results.json') -Encoding utf8
$unique = @($results.items | Sort-Object full_name -Unique)
Write-Output "UNIQUE_REPOS=$($unique.Count)"
$unique | Select-Object full_name,created_at,pushed_at,size,license,description | ConvertTo-Json -Depth 4
