<template>
  <div>
    <!-- If no more videos are available, show a thank you message -->
    <div v-if="noMoreVideos">
      <h2>Merci d'avoir voté !</h2>
      <p>Il n'y a plus de vidéos disponibles pour l'instant.</p>
    </div>
    <!-- If there are more videos, show the video and the rating buttons -->
    <div v-else>
      <!-- Display the video player if a video source is available -->
      <video v-if="videoSrc" :src="videoSrc" controls autoplay></video>
      <div class="buttons">
        <!-- Rating buttons that trigger the rateVideo method with different rating values -->
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
import axios from "axios"; // Import Axios for making HTTP requests
import { onMounted, ref } from "vue"; // Import Vue composition API hooks
import { useRouter } from "vue-router"; // Import the router for navigation

const router = useRouter();  // Initialize Vue Router
const videoSrc = ref("");  // Reference to store the video source URL
const currentVideo = ref("");  // Reference to store the current video name
const noMoreVideos = ref(false);  // Flag to track if there are no more videos
const username = ref(localStorage.getItem("username") || "");  // Retrieve the username from local storage or default to an empty string

// If the username is not set, redirect the user to the homepage
if (!username.value) {
  router.push("/");  // Navigate to the homepage
}

// Fetch the next video that the user has not rated
const fetchVideo = async () => {
  try {
    // Make a GET request to the backend to retrieve a video for the user
    const response = await axios.get(`http://127.0.0.1:5000/video?user=${username.value}`);
    if (response.data.video) {
      // If a video is returned, set the current video and video source
      currentVideo.value = response.data.video;
      videoSrc.value = `http://127.0.0.1:5000/videos/${response.data.video}`;
      noMoreVideos.value = false;  // There are videos available
    } else {
      noMoreVideos.value = true;  // No more videos to rate
    }
  } catch (error) {
    // Handle any errors during the request
    console.error("Error loading video", error);
    noMoreVideos.value = true;  // Set noMoreVideos to true in case of an error
  }
};

// Rate the current video and submit the rating
const rateVideo = async (rating: number) => {
  try {
    // Send the rating data to the backend via a POST request
    await axios.post("http://127.0.0.1:5000/rate", {
      video: currentVideo.value,  // The current video being rated
      rating: rating,  // The rating given by the user
      user: username.value,  // The username of the user
    });
    fetchVideo();  // Fetch the next video after rating
  } catch (error) {
    // Handle any errors during the request
    console.error("Error submitting rating", error);
    noMoreVideos.value = true;  // Set noMoreVideos to true in case of an error
  }
};

// Fetch the initial video when the component is mounted
onMounted(fetchVideo);
</script>

<style scoped>
/* Style the video player */
video {
  width: 100%;  /* Make the video player full width */
  max-height: 500px;  /* Set a maximum height for the video */
}

/* Style for the buttons container */
.buttons {
  margin-top: 10px;  /* Add top margin for spacing */
}

/* Style for the individual rating buttons */
button {
  margin: 5px;  /* Add margin around buttons */
  padding: 10px;  /* Add padding inside buttons */
  font-size: 20px;  /* Set the font size of the buttons */
  cursor: pointer;  /* Change cursor to pointer on hover */
}
</style>
