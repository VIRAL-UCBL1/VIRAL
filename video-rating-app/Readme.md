# Projet d'Évaluation de Vidéos

## Description
Ce projet permet d'afficher des vidéos stockées localement et de les évaluer avec une note de 1 à 5. Il utilise **Vue.js** pour le frontend et **Flask** pour le backend.

## Structure du projet
```
video-rating-app/
│── frontend/             # Application Vue.js
│   ├── index.html        # Page principale
│   ├── src/              # Code source Vue.js
│   │   ├── App.vue       # Composant principal
│   │   ├── main.ts       # Entrée de l'application
│   ├── vite.config.ts    # Configuration Vite
│   ├── package.json      # Dépendances du projet
│   ├── tsconfig.json     # Configuration TypeScript
│
│── backend/              # API Flask
│   ├── app.py            # Serveur backend
│   ├── ratings.csv       # Stockage des notes
│   ├── requirements.txt  # Dépendances Python
│   ├── videos/           # Dossier contenant les vidéos
│
│── start.sh              # Script pour démarrer l'application
│── start_wsl.sh          # Script adapté pour WSL (Windows Subsystem for Linux)
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

#### 2️⃣ Installation du frontend
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

---

# Video Rating Project

## Description
This project allows displaying locally stored videos and rating them from 1 to 5. It uses **Vue.js** for the frontend and **Flask** for the backend.

## Project Structure
```
video-rating-app/
│── frontend/             # Vue.js Application
│   ├── index.html        # Main page
│   ├── src/              # Vue.js source code
│   │   ├── App.vue       # Main component
│   │   ├── main.ts       # Application entry point
│   ├── vite.config.ts    # Vite configuration
│   ├── package.json      # Project dependencies
│   ├── tsconfig.json     # TypeScript configuration
│
│── backend/              # Flask API
│   ├── app.py            # Backend server
│   ├── ratings.csv       # Ratings storage
│   ├── requirements.txt  # Python dependencies
│   ├── videos/           # Folder containing videos
│
│── start.sh              # Script to start the application
│── start_wsl.sh          # Script for WSL (Windows Subsystem for Linux)
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

