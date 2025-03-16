import csv
import os
import random

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Autoriser toutes les requêtes


VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
CSV_FILE = "ratings.csv"

# Vérifier que le dossier vidéos existe
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)


# Charger la liste des vidéos
def get_videos():
    all_videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
    rated_videos = get_rated_videos()
    
    # Filtrer et retourner les vidéos qui n'ont pas encore été notées
    return [video for video in all_videos if video not in rated_videos]
# Récupérer une vidéo aléatoire
def get_random_video():
    videos = get_videos()
    return random.choice(videos) if videos else None

@app.route("/video", methods=["GET"])
def serve_video():
    video = get_random_video()
    if video:
        return jsonify({"video": video})
    return jsonify({"error": "No videos available"}), 404

@app.route("/rate", methods=["POST"])
def rate_video():
    data = request.json
    video_name = data.get("video")
    rating = data.get("rating")
    
    if not video_name or not rating:
        return jsonify({"error": "Invalid data"}), 400
    
    print(f"Received video: {video_name}, rating: {rating}")
    
    # Ouvrir le fichier en mode ajout (a) mais avec lecture (r) en plus
    with open(CSV_FILE, "a+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, rating])
        
        # Revenir au début du fichier pour pouvoir le lire
        file.seek(0)
        file_content = file.read()
        print("Current file content:")
        print(file_content)
    
    return jsonify({"success": True})



@app.route("/videos/<path:filename>")
def get_video_file(filename):
    print("Vidéos disponibles :", get_videos())
    return send_from_directory(VIDEO_FOLDER, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
