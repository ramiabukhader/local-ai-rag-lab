param(
    [string]$VenvDir = ".venv-demo",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-DemoCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "> $Command $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

Invoke-DemoCommand -Command $Python -Arguments @("-m", "venv", $VenvDir)
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RagIngest = Join-Path $VenvDir "Scripts\rag-ingest.exe"
$RagEvaluate = Join-Path $VenvDir "Scripts\rag-evaluate.exe"

Invoke-DemoCommand -Command $VenvPython -Arguments @("-m", "pip", "install", "-e", ".")
Invoke-DemoCommand -Command $RagIngest
Invoke-DemoCommand -Command $RagEvaluate
