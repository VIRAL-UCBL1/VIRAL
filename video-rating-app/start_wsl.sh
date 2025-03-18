#!/bin/bash

echo "🚀 Démarrage de l'application Video Rating sous WSL..."

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé. Installez-le d'abord."
    exit 1
fi

# Vérifier si Node.js est installé
if ! command -v node &> /dev/null; then
    echo "❌ Node.js n'est pas installé. Installez-le d'abord."
    exit 1
fi

# 1️⃣ Démarrage du backend Flask
echo "📡 Démarrage du serveur Flask..."
cd backend

# Charger Conda pour WSL
source ~/miniconda3/etc/profile.d/conda.sh  # Adapter ce chemin selon ton installation Conda

# Vérifier si l'environnement conda "video" existe, sinon le créer
if ! conda info --envs | grep -q "video"; then
    echo "🐍 Création de l'environnement conda 'video'..."
    conda create --name video python=3.8 -y
fi

# Activer l'environnement conda
conda activate video

# Installer les dépendances si nécessaire
pip install -r requirements.txt

# Lancer Flask en arrière-plan et enregistrer son PID
python app.py &  
FLASK_PID=$!

# Revenir à la racine du projet
cd ..

# 2️⃣ Démarrage du frontend Vue.js
echo "🌐 Démarrage du frontend Vue.js..."
cd frontend

# Vérifier si `node_modules` existe, sinon installer les dépendances
if [ ! -d "node_modules" ]; then
    echo "📦 Installation des dépendances Vue.js..."
    npm install
fi

# Lancer Vue.js en arrière-plan et enregistrer son PID
npm run dev &  
VUE_PID=$!

# Revenir à la racine du projet
cd ..

# Fonction pour arrêter les processus proprement
cleanup() {
    echo "🛑 Arrêt des applications..."

    # Arrêter Flask et Vue.js proprement
    kill $FLASK_PID
    kill $VUE_PID

    echo "✅ Applications arrêtées."
}

# Intercepter Ctrl+C et appeler cleanup
trap cleanup SIGINT

echo "✅ Tout est lancé !"
echo "🔹 Flask : http://127.0.0.1:5000"
echo "🔹 Vue.js : http://localhost:5173"

# Garder les processus en cours d'exécution jusqu'à interruption
wait
