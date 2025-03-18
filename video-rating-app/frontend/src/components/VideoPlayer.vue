<template>
  <div>
    <div v-if="noMoreVideos">
      <h2>Merci d'avoir voté !</h2>
      <p>Il n'y a plus de vidéos disponibles pour l'instant.</p>
    </div>
    <div v-else>
      <video v-if="videoSrc" :src="videoSrc" controls autoplay></video>
      <div class="buttons">
        <button @click="rateVideo(1)">⭐</button>
        <button @click="rateVideo(2)">⭐⭐</button>
        <button @click="rateVideo(3)">⭐⭐⭐</button>
        <button @click="rateVideo(4)">⭐⭐⭐⭐</button>
        <button @click="rateVideo(5)">⭐⭐⭐⭐⭐</button>
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
const noMoreVideos = ref(false);
const username = ref(localStorage.getItem("username") || "");

// Si l'utilisateur n'a pas entré son nom, on le renvoie à la page d'accueil
if (!username.value) {
  router.push("/");
}

const fetchVideo = async () => {
  try {
    const response = await axios.get(`http://127.0.0.1:5000/video?user=${username.value}`);
    if (response.data.video) {
      currentVideo.value = response.data.video;
      videoSrc.value = `http://127.0.0.1:5000/videos/${response.data.video}`;
      noMoreVideos.value = false;
    } else {
      noMoreVideos.value = true;
    }
  } catch (error) {
    console.error("Erreur lors du chargement de la vidéo", error);
    noMoreVideos.value = true;
  }
};

const rateVideo = async (rating: number) => {
  try {
    await axios.post("http://127.0.0.1:5000/rate", {
      video: currentVideo.value,
      rating: rating,
      user: username.value,
    });
    fetchVideo();  // Charge la vidéo suivante
  } catch (error) {
    console.error("Erreur lors de l'envoi de la note", error);
    noMoreVideos.value = true;
  }
};

onMounted(fetchVideo);
</script>

<style scoped>
video {
  width: 100%;
  max-height: 500px;
}
.buttons {
  margin-top: 10px;
}
button {
  margin: 5px;
  padding: 10px;
  font-size: 20px;
  cursor: pointer;
}
</style>
