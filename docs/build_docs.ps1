# PowerShell script to build the WDBX documentation

Write-Host "Building WDBX documentation..." -ForegroundColor Green

# Navigate to the docs directory
$docsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $docsDir

# Create _build directory if it doesn't exist
$buildDir = Join-Path $docsDir "_build"
$htmlDir = Join-Path $buildDir "html"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Check if the virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Virtual environment not activated. Please activate it first." -ForegroundColor Yellow
    Write-Host "Run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

# Build the documentation
try {
    sphinx-build -b html $docsDir $htmlDir

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nDocumentation built successfully!" -ForegroundColor Green
        Write-Host "Documentation is available at: $htmlDir\index.html" -ForegroundColor Green
        Write-Host "You can open it with the following command:" -ForegroundColor Green
        Write-Host "    Start-Process $htmlDir\index.html" -ForegroundColor Cyan
    } else {
        Write-Host "Error building documentation. Please check the output above." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    exit 1
} 