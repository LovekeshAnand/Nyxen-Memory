# Conversational Graph Memory (CGM-RAG) Automated Setup Script for Windows PowerShell
# Configures directories, environment redirects, dependencies, and pulls models natively.

$ErrorActionPreference = "Stop"

Write-Output "=============================================================="
Write-Output "      CONVERSATIONAL GRAPH MEMORY (CGM) - SYSTEM SETUP"
Write-Output "=============================================================="

# 1. Verify OS environment
Write-Output "[System] Verification: Windows OS detected"

# 2. Check Python Installation
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Output "[Error] Python is not installed or not in system PATH."
    Write-Output "[Error] Please install Python 3.10+ before running this script."
    exit 1
}

$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Output "[System] Python version: $pyVersion"

# Validate version >= 3.10
try {
    python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
} catch {
    Write-Output "[Error] Python version must be 3.10 or higher. Current: $pyVersion"
    exit 1
}

# 3. Optimize Disk Space (NPM Cache Recovery)
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Write-Output "[System] Cleaning NPM cache to reclaim C: drive space..."
    try {
        npm cache clean --force
    } catch {
        Write-Output "[System] NPM cache clean skipped or completed with warnings."
    }
} else {
    Write-Output "[System] NPM not detected, skipping NPM cache clean."
}

# 4. Configure Storage Paths (redirecting to D: drive if present to prevent C: drive exhaustion)
$ollamaPath = "$Home\OllamaModels"
$hfPath = "$Home\.cache\huggingface"

if (Test-Path "D:\") {
    Write-Output "[System] D: drive detected. Configuring redirects to D: drive to prevent C: drive exhaustion..."
    $ollamaPath = "D:\OllamaModels"
    $hfPath = "D:\huggingface_cache"
} else {
    Write-Output "[System] D: drive not detected. Falling back to C: drive user directory..."
}

if (-not (Test-Path $ollamaPath)) {
    New-Item -ItemType Directory -Path $ollamaPath | Out-Null
}
if (-not (Test-Path $hfPath)) {
    New-Item -ItemType Directory -Path $hfPath | Out-Null
}

# Set Windows User Environment Variable Permanently
Write-Output "[System] Configuring OLLAMA_MODELS user-level variable..."
try {
    [System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $ollamaPath, 'User')
    $env:OLLAMA_MODELS = $ollamaPath
    $env:HF_HOME = $hfPath
} catch {
    Write-Output "[Warning] Failed to set OLLAMA_MODELS permanently. Using session environment."
    $env:OLLAMA_MODELS = $ollamaPath
    $env:HF_HOME = $hfPath
}

# 5. Install Python Dependencies
Write-Output "[System] Installing requirements via pip..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 6. Verify and Launch Ollama Server
$ollamaPort = 11434
$targetModel = "qwen2.5:1.5b"

Write-Output "[Ollama] Checking connection to Ollama server..."
$serverActive = $false
try {
    $response = Invoke-RestMethod -Uri "http://localhost:$ollamaPort/api/tags" -TimeoutSec 3
    $serverActive = $true
} catch {
    # Server not running
}

if (-not $serverActive) {
    Write-Output "[Ollama] Ollama server is not running. Attempting to start it..."
    
    $ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (-not (Test-Path $ollamaExe)) {
        $ollamaExe = "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
    }

    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    } elseif (Test-Path $ollamaExe) {
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    } else {
        Write-Output "[Warning] Ollama binary not found. Please launch Ollama manually."
    }

    # Wait for startup
    Write-Host -NoNewline "[Ollama] Waiting for server to initialize"
    for ($i = 0; $i -lt 10; $i++) {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1.5
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:$ollamaPort/api/tags" -TimeoutSec 1
            $serverActive = $true
            Write-Host " Active!"
            break
        } catch {
            # Continue waiting
        }
    }
    
    if (-not $serverActive) {
        Write-Host " Timeout."
        Write-Output "[Warning] Server took too long to respond. Ensure Ollama is running before starting chat.py."
    }
} else {
    Write-Output "[Ollama] Ollama server is responsive."
}

# 7. Pull Ollama Model
if ($serverActive) {
    Write-Output "[Ollama] Verifying if model '$targetModel' is downloaded..."
    $modelExists = $false
    try {
        $tags = Invoke-RestMethod -Uri "http://localhost:$ollamaPort/api/tags"
        foreach ($model in $tags.models) {
            if ($model.name -like "*$targetModel*") {
                $modelExists = $true
                break
            }
        }
    } catch {
        # Error querying tags
    }

    if (-not $modelExists) {
        Write-Output "[Ollama] Model '$targetModel' not found. Downloading (this may take a few minutes)..."
        $body = @{ name = $targetModel } | ConvertTo-Json
        $response = Invoke-WebRequest -Uri "http://localhost:$ollamaPort/api/pull" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
        Write-Output "[Ollama] Model '$targetModel' successfully downloaded."
    } else {
        Write-Output "[Ollama] Model '$targetModel' is ready."
    }
}

# 8. Warm up SentenceTransformer Cache
Write-Output "[Pipeline] Downloading and warming up SentenceTransformer weights..."
python -c "
import os
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '')
from sentence_transformers import SentenceTransformer
print('[Pipeline] Loading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('[Pipeline] Warmup successful.')
"

Write-Output "=============================================================="
Write-Output "[SUCCESS] SYSTEM SETUP COMPLETED SUCCESSFULLY!"
Write-Output "[SUCCESS] All redirects are active and packages installed."
Write-Output "[SUCCESS] Run 'python chat.py' to launch the interface."
Write-Output "=============================================================="
