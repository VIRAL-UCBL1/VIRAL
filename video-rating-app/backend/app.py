import csv
import os
import random

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Video and rating folders
VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
RATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate")

# Ensure the existence of required folders
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(RATE_FOLDER, exist_ok=True)

# Function to retrieve unrated videos for the user
def get_videos(user):
    rated_videos = set()
    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")

    # Load already rated videos
    if os.path.exists(user_file):
        with open(user_file, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                rated_videos.add(row[0])  # Store names of rated videos

    # Retrieve available videos by environment
    available_videos = []
    for env in os.listdir(VIDEO_FOLDER):
        env_path = os.path.join(VIDEO_FOLDER, env)
        print("Environment:", env_path)  # Debug: Display environment path
        if os.path.isdir(env_path):  # Ensure it's a directory (environment)
            for video in os.listdir(env_path):
                if video.endswith(('.mp4', '.avi', '.mov', '.webm')) and video not in rated_videos:
                    available_videos.append((video, env))  # Add (video, environment)
    print("Available videos:", available_videos)  # Debug: Display available videos
    return available_videos

# Route to fetch an unrated video
@app.route("/video", methods=["GET"])
def serve_video():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "User is required"}), 400

    videos = get_videos(user)
    if not videos:
        return jsonify({"error": "No videos available to rate"}), 404

    # Randomly select a video
    video_name, environment = random.choice(videos)

    # Load environment-specific instruction text
    instruction_text = ""
    instruction_file = os.path.join(VIDEO_FOLDER, environment, "indication.txt")
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            instruction_text = f.read().strip()

    # Look for an instruction image (e.g., instruction.png or instruction.jpg)
    instruction_image = ""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = os.path.join(VIDEO_FOLDER, environment, f"instruction{ext}")
        if os.path.exists(candidate):
            instruction_image = f"http://127.0.0.1:5000/videos/{environment}/instruction{ext}"
            break

    return jsonify({
    "video": video_name,
    "environment": environment,
    "instructionText": instruction_text,
    "instructionImage": instruction_image
})


# Route to rate a video
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
    
    # Write header if file does not exist
    if not os.path.exists(user_file):
        with open(user_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "environment", "rating"])  # Add environment

    # Append rating
    with open(user_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, environment, rating])  # Store environment

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
