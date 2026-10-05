param([string[]]$Repos, [int]$ReadmeLimit = 2200)
$ErrorActionPreference = 'Stop'
$headers = @{ Accept = 'application/vnd.github+json'; 'User-Agent' = 'MegaKernel-research-round2' }
$credentialLines = "protocol=https`nhost=github.com`n`n" | git credential fill
$credential = @{}
foreach ($line in $credentialLines) {
    $split = $line.IndexOf('=')
    if ($split -gt 0) { $credential[$line.Substring(0, $split)] = $line.Substring($split + 1) }
}
if ($credential.ContainsKey('password')) { $headers.Authorization = "Bearer $($credential['password'])" }
$evidenceDir = Join-Path $PSScriptRoot 'candidate-evidence'
$null = New-Item -ItemType Directory -Path $evidenceDir -Force
foreach ($repo in $Repos) {
    try {
        $metadata = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo" -Headers $headers
        $branch = [uri]::EscapeDataString($metadata.default_branch)
        $tree = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/git/trees/${branch}?recursive=1" -Headers $headers
        $readme = ''
        try {
            $r = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/readme" -Headers $headers
            $readme = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($r.content))
        } catch { $readme = "README unavailable: $($_.Exception.Message)" }
        $commits = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/commits?since=2026-09-05T00:00:00Z&until=2026-10-06T00:00:00Z&per_page=20" -Headers $headers
        $document = [pscustomobject]@{
            repository=$repo; url=$metadata.html_url; created_at=$metadata.created_at; pushed_at=$metadata.pushed_at;
            branch=$metadata.default_branch; license=$metadata.license.spdx_id; fork=$metadata.fork; parent=$metadata.parent.full_name;
            tree_truncated=$tree.truncated;
            files=@($tree.tree | Where-Object type -eq 'blob' | Select-Object path,size);
            recent_commits=@($commits | ForEach-Object { [pscustomobject]@{date=$_.commit.committer.date; subject=($_.commit.message -split "`n")[0]; url=$_.html_url} });
            readme=$readme
        }
        $document | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $evidenceDir ($repo.Replace('/','--') + '.json')) -Encoding utf8
        "REPO=$repo FILES=$($document.files.Count) LICENSE=$($document.license)"
        $readme.Substring(0, [Math]::Min($ReadmeLimit, $readme.Length))
        'CODE_PATHS=' + (($document.files.path | Where-Object { $_ -match '\.(cu|cuh|cpp|metal|py|rs)$' } | Select-Object -First 45) -join ', ')
    } catch { "ERROR=$repo $($_.Exception.Message)" }
}
