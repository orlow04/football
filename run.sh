#!/bin/bash

# --- Configuração de Segurança ---
# 'set -e' faz o script parar imediatamente se algum comando der erro
set -e

echo "========================================================"
echo "⚽ Getafe Ball - Agente Defensivo 5v5"
echo "========================================================"

# --- 1. Configuração de Caminhos (CRÍTICO) ---
# Adiciona o diretório atual ($PWD) ao PYTHONPATH.
# Isso resolve aquele erro "ModuleNotFoundError: submission" 
# garantindo que o Python encontre a pasta 'submission' como um pacote.
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🔧 PYTHONPATH configurado para: $(pwd)"

# --- 2. Limpeza (Opcional) ---
# Remove vídeos antigos para não confundir a nova execução
if [ -d "videos_agent" ]; then
    echo "🧹 Limpando vídeos antigos..."
    rm -rf videos_agent/*.dump
fi

# --- 3. Execução do Agente ---
echo "▶️  Iniciando partida de avaliação (1000+ passos)..."

# Executa o script que nós criamos e validamos
python3 test_recording.py

echo "========================================================"
echo "✅ Execução finalizada com sucesso!"
echo "📂 Os replays (.dump) estão na pasta: videos_agent/"
echo "========================================================"