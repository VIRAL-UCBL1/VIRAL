<template>
  <!-- Bloc pseudonym visible tout le temps -->
  <div class="session-info">
          <p><strong>Your pseudonym:</strong> {{ pseudonym }}</p>
          <p><strong>Your progression:</strong> {{ score }}</p>
          <p>To resume your session later, enter this pseudonym on the homepage.</p>
        </div>
        <!-- Bloc instructions (visible uniquement si vidéos restantes) -->
    <div class="instructions" v-if="!noMoreVideos">
      <h3>Instructions:</h3>
      <p v-if="instructionText">{{ instructionText }}</p>
      <img
        v-if="instructionImage"
        :src="instructionImage"
        alt="Visual instruction"
        class="instruction-image"
      />
    </div>
  <div class="container">

    <div class="video-container">
      <!-- Bloc info vidéo et pseudonym -->
      <div class="video-header">
        <!-- Affichage info vidéo seulement si on a encore des vidéos -->
        <div class="video-meta" v-if="!noMoreVideos">
          <h3>Environment: {{ environment }}</h3>
          <video v-if="videoSrc" :src="videoSrc" controls autoplay></video>
        </div>

        
      </div>
    </div>
  </div>
  <!-- Message de fin -->
  <div v-if="noMoreVideos" class="end-message">
        <h2>Thank you for voting!</h2>
        <p>There are no more videos available at the moment.</p>
      </div>

      <!-- Bloc notation -->
      <div v-else class="rating-section">
        <p class="rating-instruction">
          Do you understand the instructions ?
        </p>
        <div class="rating-buttons">
          <p> Totally misunderstand </p>
          <button
            v-for="n in 5"
            :key="n"
            :class="{ selected: selectedUnderstand === n }"
            @click="selectedUnderstand = n"
          >
            {{ n }}
          </button>
          <p> Totally understand </p>
        </div>
        <p class="rating-instruction">
          Please rate this video to indicate whether it follows the instructions:
        </p>
        <div class="rating-buttons">
          <p> Totally unfollows </p>
          <button
            v-for="n in 5"
            :key="n"
            :class="{ selected: selectedRating === n }"
            @click="selectedRating = n"
          >
            {{ n }}
          </button>
          <p> Totally follows </p>
        </div>
        <p class="rating-instruction">
          Additional comment:
        </p>
        <div class="rating-buttons">
          <textarea class="instructions" v-model="comment" placeholder="write your comment"></textarea>
        </div>
        <button class="submit-button" @click="rateVideo">Submit</button>
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
const pseudonym = ref(localStorage.getItem("pseudonym") || "");
const selectedRating = ref(3);
const selectedUnderstand = ref(5);
const comment = ref("");
const score = ref("0/120")

const API_BASE_URL =  "https://ekoverleaf.duckdns.org"; // Base URL for the API
if (!pseudonym.value) {
  router.push("/");
}

function getCategory(str: string) {
    const categories = ['LunarLander-v3', 'highway-fast-v0', 'Hopper-v5', 'Swimmer-v5'];

    for (const category of categories) {
        if (str.startsWith(category + '-')) {
            return category;
        }
    }

    return 'Env';
}

function getsource(str: string) {
    const sources = ['text_image', 'text', 'image'];

    for (const source of sources) {
        if (str.endsWith(source)) {
            return source;
        }
    }

    return 'Videos';
}

const fetchVideo = async () => {
  try {
    const response = await axios.get(API_BASE_URL+`/video?username=${pseudonym.value}`);
    if (response.data.video) {
      currentVideo.value = response.data.video;
      environment.value = response.data.environment; // Store environment
      videoSrc.value = API_BASE_URL+`/videos/${environment.value}/${response.data.video}`;
      instructionText.value = response.data.instructionText || "";
      instructionImage.value = response.data.instructionImage || "";
      source.value = response.data.source || "videos";
      source.value = getsource(environment.value)
      environment.value = getCategory(environment.value)
    } else {
      noMoreVideos.value = true;
    }
  } catch (error) {
    console.error("Error loading video", error);
    noMoreVideos.value = true;
  }
};

const fetchScore = async () => {
  try {
    const response = await axios.get(API_BASE_URL+`/score?username=${pseudonym.value}`);
    if (response.data.score) {
      if (response.data.score < 0) {
        response.data.score += 1
      }
      score.value = String(response.data.score+"/120");
      if (response.data.score > 120){
        noMoreVideos.value = true;
      }
    }
  } catch (error) {
    console.error("Error loading video", error);
    noMoreVideos.value = true;
  }
};

const rateVideo = async () => {
  try {
    await axios.post(API_BASE_URL+"/rate", {
      video: currentVideo.value,
      environment: environment.value,
      understand: selectedUnderstand.value,
      comment: comment.value,
      rating: selectedRating.value,
      username: pseudonym.value, // <- identifiant complet ici
      source: source.value,
    });
    fetchVideo(); // Load a new video after rating
    fetchScore();
    comment.value = ""
  } catch (error) {
    console.error("Error submitting rating", error);
    noMoreVideos.value = true;
  }
};

onMounted(fetchScore);
onMounted(fetchVideo);
</script>

<style scoped>
.container {
  display: flex;
  align-items: flex-start;
  gap: 20px;
}

.instructions {
  display: inline-block;
  width: 80%;
  margin-top: 20px;
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
  width: 100%;
  display: flex;
  flex-direction: column;
  text-align: center; /* Essayez de centrer les éléments enfants */
}

.video-header {
  display: flex;
  align-items: flex-start;
  gap: 20px;
  width: 100%; /* Prend toute la largeur du conteneur vidéo */
  /* Modification ici pour gérer l'espace différemment */
}
.video-meta {
  flex: 1;
}

video {
  width: 100%;
  max-height: 500px;
}

.rating-section {
  flex-direction: column;
  align-items: center; /* Center items vertically within the rating section */
  width: 100%; /* Ensure it takes full width to center properly */
  margin-top: 20px; /* Add some space above the rating section */
  display: inline-block; /* Changez l'affichage pour que text-align fonctionne */
}

.rating-instruction {
  font-weight: bold;
  text-align: center;
  /* margin-bottom: 10px; */
  /* width: 80%; */
}

.rating-buttons {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 15px; /* Add some space below the buttons */
}

.rating-buttons button {
  padding: 10px 15px;
  font-size: 16px;
  background-color: #ddd;
  color: black;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.rating-buttons button.selected {
  background-color: #007bff;
  color: white;
}

.submit-button {
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



.session-info {
  width: auto; /* Ajuste la largeur au contenu */
  font-size: 14px;
  color: #ddd;
  background: #444;
  padding: 10px;
  border-radius: 5px;
  /* Empêche le bloc pseudonym de grandir et de prendre tout l'espace */
  flex-shrink: 0;
}


.session-info .note {
  font-style: italic;
  font-size: 12px;
  color: #bbb;
}
</style>