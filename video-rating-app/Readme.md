# Video Rating Project

## Description
This project allows displaying locally stored videos and rating them from 1 to 5. It uses **Vue.js** for the frontend and **Flask** for the backend.

## Project Structure
```
video-rating-app/
│── frontend/             # Vue.js Application (Frontend part of the app)
│   ├── index.html        # Main HTML page for the frontend
│   ├── src/              # Vue.js source code
│   │   ├── App.vue       # Main component of the application
│   │   ├── main.ts       # Entry point for the Vue.js application
│   ├── vite.config.ts    # Configuration file for Vite (build tool)
│   ├── package.json      # Dependencies and scripts for frontend
│   ├── tsconfig.json     # TypeScript configuration for the frontend
│
│── backend/              # Flask API (Backend server to handle requests)
│   ├── app.py            # Main file for running the Flask server
│   ├── requirements.txt  # Python dependencies for the backend
│   ├── rate/             # Folder to store ratings data (e.g., CSV file)
│   ├── videos/           # Folder containing the video files to be rated
│
│── start.sh              # Shell script to start the application on UNIX systems
│── start_wsl.sh          # Shell script to start the application in WSL (Windows Subsystem for Linux)
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
│── frontend/             # Application Vue.js (Partie frontend de l'application)
│   ├── index.html        # Page HTML principale pour le frontend
│   ├── src/              # Code source Vue.js
│   │   ├── App.vue       # Composant principal de l'application
│   │   ├── main.ts       # Point d'entrée pour l'application Vue.js
│   ├── vite.config.ts    # Fichier de configuration de Vite (outil de build)
│   ├── package.json      # Dépendances et scripts pour le frontend
│   ├── tsconfig.json     # Configuration TypeScript pour le frontend
│
│── backend/              # API Flask (Serveur backend pour gérer les requêtes)
│   ├── app.py            # Fichier principal pour lancer le serveur Flask
│   ├── requirements.txt  # Dépendances Python pour le backend
│   ├── rate/             # Dossier pour stocker les données de notes (par exemple, fichier CSV)
│   ├── videos/           # Dossier contenant les fichiers vidéo à évaluer
│
│── start.sh              # Script shell pour démarrer l'application sur des systèmes UNIX
│── start_wsl.sh          # Script shell pour démarrer l'application dans WSL (Windows Subsystem for Linux)

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

