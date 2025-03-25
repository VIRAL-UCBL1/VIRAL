import csv
import os
import random

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Dossiers des vidéos et des notations
VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
RATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate")

# Vérification de l'existence des dossiers
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(RATE_FOLDER, exist_ok=True)

# Fonction pour récupérer les vidéos non notées par l'utilisateur
def get_videos(user):
    rated_videos = set()
    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")

    # Charger les vidéos déjà notées
    if os.path.exists(user_file):
        with open(user_file, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # Ignorer l'en-tête
            for row in reader:
                rated_videos.add(row[0])  # Stocker le nom des vidéos déjà notées

    # Récupérer les vidéos disponibles par environnement
    available_videos = []
    for env in os.listdir(VIDEO_FOLDER):
        env_path = os.path.join(VIDEO_FOLDER, env)
        print("Environnement:", env_path)  # Debug: Afficher le chemin de l'environnement
        if os.path.isdir(env_path):  # Vérifier si c'est un dossier (environnement)
            for video in os.listdir(env_path):
                if video.endswith(('.mp4', '.avi', '.mov', '.webm')) and video not in rated_videos:
                    available_videos.append((video, env))  # Ajouter (vidéo, environnement)
    print("Vidéos disponibles:", available_videos)  # Debug: Afficher les vidéos disponibles
    return available_videos

# Route pour récupérer une vidéo non notée
@app.route("/video", methods=["GET"])
def serve_video():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "User is required"}), 400

    videos = get_videos(user)
    if not videos:
        return jsonify({"error": "No videos available to rate"}), 404

    # Sélectionner une vidéo aléatoirement
    video_name, environment = random.choice(videos)

    # Charger les instructions spécifiques à l'environnement
    instruction_file = os.path.join(VIDEO_FOLDER, environment, "indication.txt")
    instruction = "Aucune indication disponible"
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

    return jsonify({
        "video": video_name,
        "environment": environment,
        "instruction": instruction
    })

# Route pour noter une vidéo
@app.route("/rate", methods=["POST"])
def rate_video():
    data = request.json
    video_name = data.get("video")
    rating = data.get("rating")
    user = data.get("user")
    environment = data.get("environment")

    if not video_name or not rating or not user or not environment:
        return jsonify({"error": "Invalid data"}), 400

    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")
    
    # Écrire l'en-tête si le fichier n'existe pas
    if not os.path.exists(user_file):
        with open(user_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "environment", "rating"])  # Ajouter l'environnement

    # Ajouter la notation
    with open(user_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, environment, rating])  # Stocker l'environnement

    return jsonify({"success": True})

# Route pour servir une vidéo depuis un environnement
@app.route("/videos/<environment>/<filename>")
def get_video_file(environment, filename):
    video_path = os.path.join(VIDEO_FOLDER, environment, filename)
    if os.path.exists(video_path):
        return send_from_directory(os.path.join(VIDEO_FOLDER, environment), filename, as_attachment=False)
    return jsonify({"error": "Video not found"}), 404

# Lancement du serveur Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
