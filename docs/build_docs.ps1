# PowerShell script to build the WDBX documentation
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("simple", "standalone", "sphinx", "sphinx-minimal")]
    [string]$Method = "sphinx",
    
    [Parameter(Mandatory=$false)]
    [switch]$OpenAfterBuild = $false
)

Write-Host "Building WDBX documentation using $Method method..." -ForegroundColor Green

# Navigate to the docs directory
$docsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $docsDir

# Create _build directory if it doesn't exist
$buildDir = Join-Path $docsDir "_build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Set output directory based on method
$outputDir = Join-Path $buildDir $(if ($Method -eq "sphinx") { "html" } else { $Method })
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Check for required dependencies
function Test-PythonModule {
    param([string]$ModuleName)
    
    $result = python -c "import importlib.util; print(importlib.util.find_spec('$ModuleName') is not None)"
    return $result.Trim() -eq "True"
}

$requiredModules = @{
    "sphinx" = "sphinx";
    "myst-parser" = "myst-parser";
    "markdown" = "markdown";
    "pygments" = "pygments";
    "bs4" = "beautifulsoup4"
}

$missingModules = @()
foreach ($module in $requiredModules.Keys) {
    if (-not (Test-PythonModule -ModuleName $module)) {
        $missingModules += $requiredModules[$module]
    }
}

if ($missingModules.Count -gt 0) {
    Write-Host "Missing required Python modules: $($missingModules -join ', ')" -ForegroundColor Yellow
    $installChoice = Read-Host "Would you like to install them now? (y/n)"
    
    if ($installChoice -eq "y") {
        try {
            python -m pip install $missingModules
            Write-Host "Dependencies installed successfully." -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to install dependencies. Please install them manually." -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "Required dependencies missing. Build may fail." -ForegroundColor Yellow
    }
}

# Build the documentation based on method
try {
    switch ($Method) {
        "simple" {
            python build_simple.py --output $outputDir
            $indexPath = Join-Path $outputDir "index.html"
        }
        "standalone" {
            python build_docs.py --method standalone --output $outputDir
            $indexPath = Join-Path $outputDir "index.html"
        }
        "sphinx-minimal" {
            python build_minimal.py
            $indexPath = Join-Path $outputDir "index.html"
        }
        default {
            # Default sphinx build
            sphinx-build -b html $docsDir $outputDir
            $indexPath = Join-Path $outputDir "index.html"
        }
    }

    if ($LASTEXITCODE -eq 0 -and (Test-Path $indexPath)) {
        Write-Host "`nDocumentation built successfully!" -ForegroundColor Green
        Write-Host "Documentation is available at: $indexPath" -ForegroundColor Green
        
        if ($OpenAfterBuild) {
            Write-Host "Opening documentation in default browser..." -ForegroundColor Green
            Start-Process $indexPath
        }
        else {
            Write-Host "You can open it with the following command:" -ForegroundColor Green
            Write-Host "    Start-Process $indexPath" -ForegroundColor Cyan
        }
    } 
    else {
        Write-Host "Error building documentation. Please check the output above." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    exit 1
}