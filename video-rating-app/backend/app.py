import csv
import os
import random

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Initialize Flask app and enable CORS for all origins
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define folder paths for videos and ratings
VIDEO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
RATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate")

# Check if the directories for videos and ratings exist, if not, create them
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

if not os.path.exists(RATE_FOLDER):
    os.makedirs(RATE_FOLDER)

# Function to retrieve a list of videos that the user has not rated
def get_videos(user):
    # Set to store videos that the user has already rated
    rated_videos = set()
    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")
    
    # If the user has rated videos, load them into the rated_videos set
    if os.path.exists(user_file):
        with open(user_file, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                rated_videos.add(row[0])  # Add video name to the set of rated videos
    
    # Get all videos in the VIDEO_FOLDER that are not rated by the user
    available_videos = [
        f for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(('.mp4', '.avi', '.mov', '.webm')) and f not in rated_videos
    ]
    
    return available_videos

# Route to serve a random video that the user has not rated
@app.route("/video", methods=["GET"])
def serve_video():
    user = request.args.get("user")  # Get the user from the query parameters
    
    # If no user is provided, return an error
    if not user:
        return jsonify({"error": "User is required"}), 400
    
    videos = get_videos(user)  # Get list of un-rated videos
    
    instruction_file = os.path.join(VIDEO_FOLDER, "indication.txt")

    instruction = "Aucune indication disponible"
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            instruction = f.read().strip()
    
    # If there are available videos, return one at random
    if videos:
        return jsonify({"video": random.choice(videos), "instruction": instruction})

            
        
    return jsonify({"error": "No videos available to rate"}), 404  # If no videos are available, return an error

# Route to handle the rating of a video by the user
@app.route("/rate", methods=["POST"])
def rate_video():
    data = request.json
    video_name = data.get("video")  # Get video name from the request data
    rating = data.get("rating")  # Get rating from the request data
    user = data.get("user")  # Get user from the request data

    # Validate that all required data is present
    if not video_name or not rating or not user:
        return jsonify({"error": "Invalid data"}), 400

    # File name for storing user ratings, based on the user's name
    user_file = os.path.join(RATE_FOLDER, f"{user}.csv")

    # If the file does not exist, create it and write headers
    if not os.path.exists(user_file):
        with open(user_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "rating"])  # Write headers if the file is new

    # Append the new rating to the user's file
    with open(user_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([video_name, rating])  # Write the video name and rating

    return jsonify({"success": True})  # Return success response

# Route to serve the video file itself based on the filename
@app.route("/videos/<path:filename>")
def get_video_file(filename):
    video_path = os.path.join(VIDEO_FOLDER, filename)  # Get the full path to the video
    if os.path.exists(video_path):  # If the video exists, return it
        return send_from_directory(VIDEO_FOLDER, filename, as_attachment=False)
    return jsonify({"error": "Video not found"}), 404  # If the video doesn't exist, return an error

# Start the Flask app when the script is run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
