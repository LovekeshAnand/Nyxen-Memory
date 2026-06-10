#!/usr/bin/env bash

# Conversational Graph Memory (CGM-RAG) Automated Setup Script
# Configures directories, environment redirects, dependencies, and pulls models.

set -euo pipefail

echo "=============================================================="
echo "      CONVERSATIONAL GRAPH MEMORY (CGM) - SYSTEM SETUP"
echo "=============================================================="

# 1. Detect Operating System and Architecture
OS_TYPE="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS_TYPE="windows"
fi
echo "[System] Detected OS: $OS_TYPE"

# 2. Check Python Installation
if ! command -v python &> /dev/null; then
    echo "[Error] Python is not installed or not in system PATH."
    echo "[Error] Please install Python 3.10+ before running this script."
    exit 1
fi

PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[System] Python version: $PY_VERSION"
# Validate version >= 3.10
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo "[Error] Python version must be 3.10 or higher. Current: $PY_VERSION"
    exit 1
}

# 3. Optimize Disk Space (NPM Cache Recovery)
if command -v npm &> /dev/null; then
    echo "[System] Cleaning NPM cache to reclaim C: drive space..."
    npm cache clean --force || true
else
    echo "[System] NPM not detected, skipping NPM cache clean."
fi

# 4. Configure Storage Paths (redirecting to D: drive if present to prevent C: drive exhaustion)
if [[ "$OS_TYPE" == "windows" ]]; then
    IS_D_LOCAL_FIXED=false
    if [ -d "/d" ] || [ -d "D:\\" ] || [ -d "D:/" ]; then
        if command -v powershell.exe &> /dev/null; then
            D_TYPE=$(powershell.exe -Command "(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='D:'\" -ErrorAction SilentlyContinue).DriveType" 2>/dev/null | tr -d '\r\n[:space:]')
            if [ -z "$D_TYPE" ]; then
                D_TYPE=$(powershell.exe -Command "(Get-WmiObject Win32_LogicalDisk -Filter \"DeviceID='D:'\" -ErrorAction SilentlyContinue).DriveType" 2>/dev/null | tr -d '\r\n[:space:]')
            fi
            if [ "$D_TYPE" = "3" ]; then
                IS_D_LOCAL_FIXED=true
            else
                echo "[System] D: drive detected but it is not a local fixed disk (DriveType: ${D_TYPE:-Unknown}). Skipping redirect to avoid network share bottlenecks."
            fi
        else
            echo "[System] D: drive detected, but powershell.exe not found to verify drive type. Skipping redirect."
        fi
    fi

    if [ "$IS_D_LOCAL_FIXED" = true ]; then
        echo "[System] Local fixed D: drive detected. Configuring redirects to D: drive to prevent C: drive exhaustion..."
        OLLAMA_TARGET="D:\\OllamaModels"
        HF_TARGET="D:\\huggingface_cache"
    else
        echo "[System] Using default user directory storage on C: drive..."
        OLLAMA_TARGET="$HOME\\OllamaModels"
        HF_TARGET="$HOME\\.cache\\huggingface"
    fi
    
    mkdir -p "${OLLAMA_TARGET//\\//}"
    mkdir -p "${HF_TARGET//\\//}"
    
    # Set Windows user environment variable permanently using setx
    echo "[System] Configuring OLLAMA_MODELS user-level variable..."
    setx OLLAMA_MODELS "$OLLAMA_TARGET" > /dev/null || echo "[Warning] Failed to set OLLAMA_MODELS permanently. Using session environment."
    export OLLAMA_MODELS="$OLLAMA_TARGET"
    export HF_HOME="$HF_TARGET"
else
    # Linux/macOS fallback paths
    echo "[System] Setting up storage directories in user home directory..."
    mkdir -p "$HOME/OllamaModels"
    mkdir -p "$HOME/huggingface_cache"
    export OLLAMA_MODELS="$HOME/OllamaModels"
    export HF_HOME="$HOME/huggingface_cache"
fi

# 5. Install Python Dependencies
echo "[System] Installing requirements via pip..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 6. Verify and Launch Ollama Server
OLLAMA_PORT=11434
TARGET_MODEL="qwen2.5:1.5b"

echo "[Ollama] Checking connection to Ollama server..."
SERVER_ACTIVE=false
if curl -s -o /dev/null -w "%{http_code}" http://localhost:${OLLAMA_PORT}/api/tags &> /dev/null; then
    SERVER_ACTIVE=true
fi

if [ "$SERVER_ACTIVE" = false ]; then
    echo "[Ollama] Ollama server is not running. Attempting to start it..."
    if [[ "$OS_TYPE" == "windows" ]]; then
        # Default Windows Ollama EXE location
        OLLAMA_EXE="/c/Users/$USER/AppData/Local/Programs/Ollama/ollama.exe"
        if [ ! -f "$OLLAMA_EXE" ]; then
            OLLAMA_EXE="C:\\Users\\$USER\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
        fi
        
        if command -v ollama &> /dev/null; then
            ollama serve &> /dev/null &
        elif [ -f "$OLLAMA_EXE" ]; then
            "$OLLAMA_EXE" serve &> /dev/null &
        else
            echo "[Warning] Ollama binary not found at default location. Please launch Ollama manually."
        fi
    else
        if command -v ollama &> /dev/null; then
            ollama serve &> /dev/null &
        fi
    fi
    
    # Wait for startup
    echo -n "[Ollama] Waiting for server to initialize"
    for i in {1..10}; do
        echo -n "."
        sleep 1.5
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:${OLLAMA_PORT}/api/tags &> /dev/null; then
            SERVER_ACTIVE=true
            echo " Active!"
            break
        fi
    done
    if [ "$SERVER_ACTIVE" = false ]; then
        echo " Timeout."
        echo "[Warning] Server took too long to respond. Ensure Ollama is running before starting chat.py."
    fi
else
    echo "[Ollama] Ollama server is responsive."
fi

# 7. Pull Ollama Model
if [ "$SERVER_ACTIVE" = true ]; then
    echo "[Ollama] Verifying if model '$TARGET_MODEL' is downloaded..."
    MODEL_EXISTS=$(curl -s http://localhost:${OLLAMA_PORT}/api/tags | grep -q "$TARGET_MODEL" && echo "true" || echo "false")
    
    if [ "$MODEL_EXISTS" = "false" ]; then
        echo "[Ollama] Model '$TARGET_MODEL' not found. Downloading (this may take a few minutes)..."
        curl -d "{\"name\":\"$TARGET_MODEL\"}" http://localhost:${OLLAMA_PORT}/api/pull
        echo ""
        echo "[Ollama] Model '$TARGET_MODEL' successfully downloaded."
    else
        echo "[Ollama] Model '$TARGET_MODEL' is ready."
    fi
fi

# 8. Compile Rust Speedups Library
echo "[System] Checking for Rust/Cargo installation..."
if command -v cargo &> /dev/null; then
    echo "[System] Cargo detected. Compiling Rust speedups library from source..."
    (
        cd "$(dirname "$0")/cgm_rust_speedups"
        cargo build --release
    )
    
    # Copy based on OS
    if [[ "$OS_TYPE" == "windows" ]]; then
        DLL_SRC="$(dirname "$0")/cgm_rust_speedups/target/release/cgm_rust_speedups.dll"
        DLL_DST="$(dirname "$0")/cgm/safety/cgm_rust_speedups.dll"
    elif [[ "$OS_TYPE" == "macos" ]]; then
        DLL_SRC="$(dirname "$0")/cgm_rust_speedups/target/release/libcgm_rust_speedups.dylib"
        DLL_DST="$(dirname "$0")/cgm/safety/libcgm_rust_speedups.dylib"
    else
        DLL_SRC="$(dirname "$0")/cgm_rust_speedups/target/release/libcgm_rust_speedups.so"
        DLL_DST="$(dirname "$0")/cgm/safety/libcgm_rust_speedups.so"
    fi
    
    if [ -f "$DLL_SRC" ]; then
        mkdir -p "$(dirname "$0")/cgm/safety"
        cp "$DLL_SRC" "$DLL_DST"
        echo "[System] Successfully compiled and deployed: $DLL_DST"
    else
        echo "[Warning] Compiled binary not found at: $DLL_SRC"
    fi
else
    echo "[System] Cargo not found. Skipping Rust compilation; the pipeline will fall back to pure Python implementations."
fi

# 9. Warm up SentenceTransformer Cache
echo "[Pipeline] Downloading and warming up SentenceTransformer weights..."
python -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'
from sentence_transformers import SentenceTransformer
print('[Pipeline] Loading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('[Pipeline] Warmup successful.')
"

echo "=============================================================="
echo "[SUCCESS] SYSTEM SETUP COMPLETED SUCCESSFULLY!"
echo "[SUCCESS] All redirects are active and packages installed."
echo "[SUCCESS] Run 'python chat.py' to launch the interface."
echo "=============================================================="
