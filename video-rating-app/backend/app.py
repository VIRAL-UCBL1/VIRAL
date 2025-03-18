import csv
import os
import random

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
RATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate")

# Vérifier que les dossiers existent
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

if not os.path.exists(RATE_FOLDER):
    os.makedirs(RATE_FOLDER)

# Charger la liste des vidéos
def get_videos(user):
    # Charger les vidéos déjà notées par l'utilisateur
    rated_videos = set()
    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")
    
    if os.path.exists(user_file):
        with open(user_file, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                rated_videos.add(row[0])  # Ajouter la vidéo à l'ensemble des vidéos notées
    
    # Récupérer les vidéos qui n'ont pas encore été notées par l'utilisateur
    available_videos = [
        f for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(('.mp4', '.avi', '.mov', '.webm')) and f not in rated_videos
    ]
    
    return available_videos

@app.route("/video", methods=["GET"])
def serve_video():
    user = request.args.get("user")  # Récupérer l'utilisateur via les paramètres de la requête
    
    if not user:
        return jsonify({"error": "User is required"}), 400
    
    videos = get_videos(user)
    
    if videos:
        return jsonify({"video": random.choice(videos)})  # Renvoyer une vidéo non notée
    return jsonify({"error": "No videos available to rate"}), 404

@app.route("/rate", methods=["POST"])
def rate_video():
    data = request.json
    video_name = data.get("video")
    rating = data.get("rating")
    user = data.get("user")

    if not video_name or not rating or not user:
        return jsonify({"error": "Invalid data"}), 400

    # Nom du fichier basé sur l'utilisateur
    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")

    # Créer le fichier s'il n'existe pas
    if not os.path.exists(user_file):
        with open(user_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "rating"])  # Ajouter les en-têtes si nécessaire

    with open(user_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, rating])  # Sauvegarder la note

    return jsonify({"success": True})

@app.route("/videos/<path:filename>")
def get_video_file(filename):
    video_path = os.path.join(VIDEO_FOLDER, filename)
    if os.path.exists(video_path):
        return send_from_directory(VIDEO_FOLDER, filename, as_attachment=False)
    return jsonify({"error": "Video not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
