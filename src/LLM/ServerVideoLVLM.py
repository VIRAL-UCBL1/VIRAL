from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN
)
from llava.conversation import conv_templates
import copy
import torch
import os
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")
import subprocess
import re

from flask import Flask, request, jsonify
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'video'


def _execute_ollama_ps():
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, check=True
        )
        output = result.stdout
        print("stdout 'ollama ps':")
        print(output)

        # Utilisation d'une expression régulière pour extraire le premier nom
        match = re.search(
            r"NAME\s+ID\s+SIZE\s+PROCESSOR\s+UNTIL\s+(.+?)\s+", output, re.DOTALL
        )

        if match:
            first_name = match.group(1).strip()
            print(f"ollama model : {first_name}")
            return first_name
        else:
            print("ollama ps return None")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error : {e}")
        print(f"Stderr: {e.stderr}")
        return None


def _execute_ollama_stop(model: str):
    try:
        result = subprocess.run(
            ["ollama", "stop", model], capture_output=True, text=True, check=True
        )
        output = result.stdout
        print("stdout 'ollama stop':")
        print(output)

    except subprocess.CalledProcessError as e:
        print(f"Error : {e}")
        print(f"Stderr: {e.stderr}")

# Function to load video and sample frames
def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))

    # Create a VideoReader object to read the video
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    # Get frames in the video
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()

    # Adjust the sampling rate based on the desired fps
    fps = round(vr.get_avg_fps() / fps)

    # Create a list of frame indices to sample from the video
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]  # the time for each sampled frame

    if len(frame_idx) > max_frames_num or force_sample:
        # Sample uniformly from the total frames
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, sample_fps, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    # Return the sampled frames, their timestamps, and the total video duration
    return spare_frames, frame_time, video_time


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'Video not in the request'
    file = request.files['file']
    filename = 'tmp.mp4'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print('upload_video filename: ' + filename)
    return 'Video successfully uploaded'

@app.route("/", methods=["POST"])
def process_video():
    # Load the pretrained model and tokenizer
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
    )
    model.eval()
   
    data = request.get_json()
    video_path = './video/tmp.mp4'
    question = data['prompt']
    max_frames_num = 64 # max 110
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
    video = [video]

    conv_template = "qwen_1_5"
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
    
    full_question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{question}"
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], full_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0.6,
                max_new_tokens=4096,
            )
    except Exception:
        print('error mem cuda, going to free one model of ollama')
        ollama_model = _execute_ollama_ps()
        _execute_ollama_stop(ollama_model)
        ollama_model = _execute_ollama_ps()

        torch.cuda.empty_cache()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0.6,
                max_new_tokens=4096,
            )


    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
    response = {'response': response}
    del tokenizer
    del model
    del image_processor
    del max_length
    del video
    del input_ids
    torch.cuda.empty_cache()
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
