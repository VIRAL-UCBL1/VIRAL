<template>
  <div class="container">
    <!-- Instructions Section -->
    <div class="instructions" v-if="!noMoreVideos">
      <h3>Instructions :</h3>
      <p v-if="instructionText">{{ instructionText }}</p>
      <img
        v-if="instructionImage"
        :src="instructionImage"
        alt="Instruction visuelle"
        class="instruction-image"
      />
    </div>
    <!-- Video Player Section -->
    <div class="video-container">
      <div v-if="noMoreVideos">
        <h2>Thank you for voting!</h2>
        <p>There are no more videos available at the moment.</p>
      </div>
      <div v-else>
        <h3>Environment: {{ environment }}</h3> <!-- Displaying environment -->
        <video v-if="videoSrc" :src="videoSrc" controls autoplay></video>

        <div class="rating-container">
          <p class="rating-instruction">
            Veuillez donner une note à cette vidéo pour indiquer si celle-ci respecte les consignes affichées à gauche de la vidéo :
          </p>
          <div class="rating-buttons">
            <button
              v-for="n in 5"
              :key="n"
              :class="{ selected: selectedRating === n }"
              @click="selectedRating = n"
            >
              {{ n }}
            </button>
          </div>
          <button class="submit-button" @click="rateVideo">Soumettre</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import axios from "axios";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const videoSrc = ref("");
const currentVideo = ref("");
const environment = ref(""); // Stores the video environment
const noMoreVideos = ref(false);
const instructionText = ref(""); // Stores environment instructions
const instructionImage = ref(""); // Stores environment image
const source = ref("videos"); // par défaut
const username = ref(localStorage.getItem("username") || "");
const selectedRating = ref(3);

if (!username.value) {
  router.push("/");
}

const fetchVideo = async () => {
  try {
    const response = await axios.get(`http://127.0.0.1:5000/video?user=${username.value}`);
    if (response.data.video) {
      currentVideo.value = response.data.video;
      environment.value = response.data.environment; // Store environment
      videoSrc.value = `http://127.0.0.1:5000/videos/${environment.value}/${response.data.video}`;
      instructionText.value = response.data.instructionText || "";
      instructionImage.value = response.data.instructionImage || "";
      source.value = response.data.source || "videos";
      noMoreVideos.value = false;
    } else {
      noMoreVideos.value = true;
    }
  } catch (error) {
    console.error("Error loading video", error);
    noMoreVideos.value = true;
  }
};

const rateVideo = async () => {
  try {
    await axios.post("http://127.0.0.1:5000/rate", {
      video: currentVideo.value,
      environment: environment.value,
      rating: selectedRating.value,
      user: username.value,
      source: source.value,
    });
    fetchVideo(); // Load a new video after rating
  } catch (error) {
    console.error("Error submitting rating", error);
    noMoreVideos.value = true;
  }
};

onMounted(fetchVideo);
</script>

<style scoped>
.container {
  display: flex;
  align-items: flex-start;
  gap: 20px;
}

.instructions {
  width: 30%;
  background: #333;
  color: white;
  padding: 15px;
  border-radius: 5px;
}

.instruction-image {
  width: 100%;
  max-height: 300px;
  object-fit: contain;
  margin-top: 10px;
  border: 1px solid #555;
  border-radius: 5px;
}


.video-container {
  width: 70%;
}

video {
  width: 100%;
  max-height: 500px;
}

.rating-instruction {
  font-weight: bold;
  text-align: center;
  margin-bottom: 10px;
}

.rating-buttons {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.rating-buttons button {
  padding: 10px 15px;
  font-size: 16px;
  background-color: #ddd;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.rating-buttons button.selected {
  background-color: #007bff;
  color: white;
}

.submit-button {
  margin-top: 15px;
  padding: 10px 20px;
  font-size: 16px;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.submit-button:hover {
  background-color: #218838;
}


input[type="range"] {
  width: 80%;
  margin-top: 10px;
}

button {
  margin-top: 10px;
  padding: 10px 15px;
  font-size: 16px;
  cursor: pointer;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
}

button:hover {
  background-color: #0056b3;
}
</style>
