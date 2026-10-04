$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\run_swing.py" @args
if ($LASTEXITCODE -ne 0) { throw 'Swing scan did not finish. Previous saved results remain available.' }
