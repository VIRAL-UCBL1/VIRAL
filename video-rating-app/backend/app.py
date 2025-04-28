import csv
import os
import pandas as pd
import random

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Video and rating folders
VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
RATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate")
VALIDATION_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation")
API_BASE_URL = "https://ekoverleaf.duckdns.org/"
# Ensure the existence of required folders
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(RATE_FOLDER, exist_ok=True)
os.makedirs(VALIDATION_FOLDER, exist_ok=True)



# # # Function to retrieve unrated videos for the user
# def get_videos(user):
#     rated_videos = set()
#     user_file = os.path.join(RATE_FOLDER, f"{user}.csv")
#     # Lire les vidéos déjà notées (avec leur source)
#     if os.path.exists(user_file):
#         with open(user_file, "r", newline="") as file:
#             reader = csv.reader(file)
#             next(reader, None)
#             for row in reader:
#                 # row[0] = video name, row[1] = environment, row[3] = source
#                 rated_videos.add((row[0], row[1], row[3] if len(row) > 3 else "videos"))
                
#     def collect_unrated(folder, source_name):
#         available = []
#         for env in os.listdir(folder):
#             env_path = os.path.join(folder, env)
#             if os.path.isdir(env_path):
#                 for video in os.listdir(env_path):
#                     if video.endswith(('.mp4', '.avi', '.mov', '.webm')):
#                         key = (video, env, source_name)
#                         if key not in rated_videos:
#                             available.append((video, env, source_name))
#         return available

#     # Priorité : validation d'abord
#     validation_videos = collect_unrated(VALIDATION_FOLDER, "validation")
#     if validation_videos:
#         return validation_videos

#     # Sinon, on retourne les vidéos normales
#     return collect_unrated(VIDEO_FOLDER, "videos")

def get_videos(username):
    dir_video = []
    
    for environment in os.listdir(VIDEO_FOLDER):
        dir_video.append(environment)
    env_dir_count = {v: 0 for v in dir_video}
    
    for csv_file in os.listdir(RATE_FOLDER):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(RATE_FOLDER, csv_file)
            try:
                with open(csv_path, 'r', newline='') as file:
                    reader = csv.reader(file)
                    header = next(reader, None) 
                    if not header: 
                        continue
                    for row in reader:
                        key = row[1] + '-' + row[5]  # environment, source
                        if key in env_dir_count:
                            env_dir_count[key] += 1
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

    min_rating_count = min(env_dir_count.values()) if env_dir_count else 0
    selected_env = [v for v in dir_video if env_dir_count[v] == min_rating_count]
    final_choice = random.choices(selected_env)[0]
    print(final_choice, type(final_choice))
    env_path = os.path.join(VIDEO_FOLDER, final_choice)
    video_to_rate = set()
    for v_file in os.listdir(env_path):
        if v_file.endswith('.mp4'):
            video_to_rate.add(v_file)
    
    # lire mon csv user et select que les trucs qui sont dans mon final_choice c'est les vidéo
    user_file = os.path.join(RATE_FOLDER, f"{username}.csv")
    if not os.path.exists(user_file):
        with open(user_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "environment", "understand", "comment", "rating", "source"])  # Add environment
    
    data = pd.read_csv(user_file)
    data = data[data['environment'].isin([final_choice])]
    data = data[data['source'].isin([final_choice])]
    print(data)
    video_to_rate = video_to_rate - set(data['video_name'])
    res = random.choices(list(video_to_rate))[0]
    print(res)
    return res, final_choice

    # for csv_file in os.listdir(RATE_FOLDER):
    #     if csv_file == f'{username}.csv':
    #         csv_path = os.path.join(RATE_FOLDER, csv_file)
    #         with open(csv_path, 'r', newline='') as file:
    #             reader = csv.reader(file)
    #             header = next(reader, None) 
    #             if not header: 
    #                 continue
    #             for row in reader:
    # if os.path.isdir(env_path):
    #     for file in os.listdir(env_path):
    #         if file.lower().endswith('.mp4'):
    #             # ET QUI NE SONT PAS DÉJA RATE APPEND DANS LIST ET RETURN RANDOM CHOICE
    #             videos.append((file, environment, "videos"))
    # Get videos from the VALIDATION_FOLDER
    # for environment in os.listdir(VALIDATION_FOLDER):
    #     env_path = os.path.join(VALIDATION_FOLDER, environment)
    #     if os.path.isdir(env_path):
    #         for file in os.listdir(env_path):
    #             if file.lower().endswith('.mp4'):
    #                 videos.append((file, environment, "validation"))
    
    
    # Get videos that have been rated the least number of times
    
    # return least_rated_videos


# Route to fetch an unrated video
@app.route("/video", methods=["GET"])
def serve_video():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "username is required"}), 400

    videos, environment = get_videos(username)
    if not videos:
        return jsonify({"error": "No videos available to rate"}), 404

    # Randomly select a video
    # video_name, environment, source = random.choice(videos)

    base_folder = VIDEO_FOLDER # VALIDATION_FOLDER if source == "validation" else 

    # Texte
    instruction_text = ""
    instruction_file = os.path.join(base_folder, environment, "indication.txt")
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            instruction_text = f.read().strip()

    # Image
    instruction_image = ""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = os.path.join(base_folder, environment, f"instruction{ext}")
        if os.path.exists(candidate):
            instruction_image = f"{API_BASE_URL}/videos/{environment}/instruction{ext}"
            break

    return jsonify({
    "video": videos,
    "environment": environment,
    "instructionText": instruction_text,
    "instructionImage": instruction_image,
    "source": ""
})


@app.route("/score", methods=["GET"])
def get_score():
    username = request.args.get("username")
    user_file = os.path.join(RATE_FOLDER, f"{username}.csv")
    line_count = 0
    if os.path.exists(user_file):
        with open(user_file, "r") as file:
            line_count = sum(1 for line in file)
    return jsonify({"score": line_count-1})


# Route to rate a video
@app.route("/rate", methods=["POST"])
def rate_video():
    data = request.json
    video_name = data.get("video")
    rating = data.get("rating")
    understand = data.get("understand")
    comment = data.get("comment")
    username = data.get("username")
    source = data.get("source", "videos")
    environment = data.get("environment")

    if not video_name or rating is None or not username or not environment:
        return jsonify({"error": "Invalid data"}), 400

    user_file = os.path.join(RATE_FOLDER, f"{username}.csv")

    # Write header if file does not exist
    if not os.path.exists(user_file):
        with open(user_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "environment", "understand", "comment", "rating", "source"])  # Add environment

    # Append rating
    with open(user_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, environment, understand, comment, rating, source])  # Store environment

    return jsonify({"success": True})

# Route to serve a video from an environment
@app.route("/videos/<environment>/<filename>")
def get_video_file(environment, filename):
    video_path = os.path.join(VIDEO_FOLDER, environment, filename)
    if os.path.exists(video_path):
        return send_from_directory(os.path.join(VIDEO_FOLDER, environment), filename, as_attachment=False)
    return jsonify({"error": "Video not found"}), 404

# Start the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
