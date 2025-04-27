# Video Rating Project

## Description
This project allows displaying locally stored videos and rating them from 1 to 5. It uses **Vue.js** for the frontend and **Flask** for the backend.

## Project Structure
```
video-rating-app/
│
├── backend/                      # Flask API backend
│   ├── app.py                    # Main entry point for the Flask server
│   ├── requirements.txt          # Python dependencies
│   ├── rate/                     # Rating data (CSV files are ignored)
│   ├── validation/               # Data validation or processing logic
│   └── videos/                   # Folder for video files (present but ignored)
│
├── frontend/                     # Vue.js frontend application
│   ├── index.html                # Main HTML page
│   ├── public/                   # Static public files
│   ├── src/                      # Vue.js source code (components, views, etc.)
│   ├── vite.config.ts            # Vite build tool configuration
│   ├── package.json              # Frontend dependencies & scripts
│   ├── tsconfig.json             # Base TypeScript configuration
│   ├── tsconfig.app.json         # App-specific TS config
│   ├── tsconfig.node.json        # TS config for Node-based tooling
│   └── README.md                 # Frontend app documentation
│
├── .gitignore                   # Git ignore rules
├── package.json                 # Global scripts or dependencies (if any)
├── Readme.md                    # Main project documentation
├── start.sh                     # Shell script to start the app (Unix)
└── start_wsl.sh                 # Script to start the app on WSL
```


## Installation and Running
### Prerequisites
- **Node.js** and **npm** (or yarn) for the frontend
- **Python 3** and **pip** for the backend

### Installation
#### 1️⃣ Install the backend
```sh
cd backend
python -m venv venv  # Create virtual environment
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
pip install -r requirements.txt  # Install dependencies
```

#### 2️⃣ Install the frontend
```sh
cd frontend
npm install  # Install Vue.js dependencies
```

### Run the project
#### Method 1: Use the automatic script
From the project root, run:
```sh
./start.sh  # For Linux/Mac
./start_wsl.sh  # For WSL
```

#### Method 2: Start manually
In the first terminal, start the backend:
```sh
cd backend
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
python app.py
```

In a second terminal, start the frontend:
```sh
cd frontend
npm run dev
```

---

# Projet d'Évaluation de Vidéos

## Description
Ce projet permet d'afficher des vidéos stockées localement et de les évaluer avec une note de 1 à 5. Il utilise **Vue.js** pour le frontend et **Flask** pour le backend.

## Structure du Projet
```
video-rating-app/
│
├── backend/                      # Backend Flask API
│   ├── app.py                    # Point d'entrée du serveur Flask
│   ├── requirements.txt          # Dépendances Python
│   ├── rate/                     # Données de notation (sans les fichiers CSV)
│   ├── validation/               # Logique de validation ou traitement des données
│   └── videos/                   # Vidéos à noter (présent dans l’arborescence mais ignoré)
│
├── frontend/                     # Application Vue.js (frontend)
│   ├── index.html                # Page HTML principale
│   ├── public/                   # Fichiers statiques publics
│   ├── src/                      # Code source Vue.js
│   ├── vite.config.ts            # Configuration Vite
│   ├── package.json              # Dépendances & scripts frontend
│   ├── tsconfig.json             # Config de base TypeScript
│   ├── tsconfig.app.json         # Config TS spécifique à l’app
│   ├── tsconfig.node.json        # Config TS pour outils côté Node
│   └── README.md                 # Infos sur l'application frontend
│
├── .gitignore                   # Liste des fichiers/dossiers à ignorer par Git
├── package.json                 # Scripts ou dépendances globales
├── Readme.md                    # Documentation principale du projet
├── start.sh                     # Script de démarrage Unix
└── start_wsl.sh                 # Script de démarrage pour WSL

```

## Installation et Lancement
### Prérequis
- **Node.js** et **npm** (ou yarn) pour le frontend
- **Python 3** et **pip** pour le backend

### Installation
#### 1️⃣ Installation du backend
```sh
cd backend
python -m venv venv  # Création d'un environnement virtuel
source venv/bin/activate  # (Sous Windows: venv\Scripts\activate)
pip install -r requirements.txt  # Installation des dépendances
```

#### Installation du frontend
```sh
cd frontend
npm install  # Installation des dépendances Vue.js
```

### Lancer le projet
#### Méthode 1 : Utiliser le script automatique
Depuis la racine du projet, exécutez :
```sh
./start.sh  # Pour Linux/Mac
./start_wsl.sh  # Pour WSL
```

#### Méthode 2 : Lancer manuellement
Dans un premier terminal, démarrez le backend :
```sh
cd backend
source venv/bin/activate  # (Sous Windows: venv\Scripts\activate)
python app.py
```

Dans un second terminal, démarrez le frontend :
```sh
cd frontend
npm run dev
```

