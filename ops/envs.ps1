<#
.SYNOPSIS
    Per-stage virtual environments, one per workspace package.

.DESCRIPTION
    The default .venv holds everything — torch AND tensorflow AND mediapipe AND
    opencv — because one workspace root pulls in every member. That is convenient
    and it is also where dependency conflicts come from: torch and tensorflow both
    pin numpy and protobuf ranges, and `uv sync` resolves them together or not at
    all.

    A per-stage env is `uv sync --package <member>` with UV_PROJECT_ENVIRONMENT
    pointed somewhere else. Each one installs exactly that package's dependency
    closure, so a stage cannot accidentally import something it never declared.

    Verified isolation (uv sync --package sb-mlops --dry-run, 2026-09-05):
    an mlops-only environment drops torch, tensorflow, mediapipe and opencv —
    which is the `sb-mlops must not import sb-recognize` invariant made
    executable rather than merely documented.

    Extraction is the stage that most wanted its own env, and it is leaving
    Python entirely for Deno (packages/sb-extract-ts), so the split that remains
    worth making is training (torch) vs registry (neither) vs export (tensorflow).

.PARAMETER Stage
    train | mlops | extract | all. Default: all.

.EXAMPLE
    ./ops/envs.ps1 -Stage train
    .venvs/train/Scripts/python.exe -c "import torch; print(torch.__version__)"

.NOTES
    The default .venv is left alone. Notebooks and the console scripts keep using
    it; these are for running one stage in isolation, or for a machine that only
    needs one of them.
#>
param(
    [ValidateSet('train', 'mlops', 'extract', 'all')]
    [string]$Stage = 'all'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

# stage -> workspace member whose dependency closure defines the environment
$members = @{
    train   = 'sb-recognize'
    mlops   = 'sb-mlops'
    extract = 'sb-extract'
}

$targets = if ($Stage -eq 'all') { $members.Keys } else { @($Stage) }

foreach ($name in $targets) {
    $member = $members[$name]
    $envPath = Join-Path $repo ".venvs/$name"
    Write-Host "==> $name  ($member)  ->  $envPath" -ForegroundColor Cyan

    # UV_PROJECT_ENVIRONMENT is what keeps this out of the default .venv
    $env:UV_PROJECT_ENVIRONMENT = $envPath
    try {
        & uv sync --package $member
        if ($LASTEXITCODE -ne 0) { throw "uv sync --package $member failed" }
    }
    finally {
        Remove-Item Env:\UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    }
}

Write-Host ''
Write-Host 'Done. The default .venv is unchanged.' -ForegroundColor Green
Write-Host 'Check the isolation holds:' -ForegroundColor Green
Write-Host '  .venvs/mlops/Scripts/python.exe -c "import sb.mlops.registry; import torch"'
Write-Host '  # ^ should fail on torch: the registry must be readable without it'
