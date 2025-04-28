<template>
  <div class="container">
    <h2>Welcome!</h2>
    <p>Enter your pseudonym</p>

    <input v-model="pseudonym" type="text" placeholder="Pseudonym" />

    <button @click="startRating">Start</button>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const pseudonym = ref("");

const startRating = () => {
  if (pseudonym.value.trim()) {

    const username = `${pseudonym.value.trim()}`;

    // Stockage en localStorage
    localStorage.setItem("pseudonym", pseudonym.value.trim());

    // Rediriger vers la page de notation
    router.push("/rate");
  } else {
    alert("Veuillez entrer votre pseudonym !");
  }
};
</script>

<style scoped>
.container {
  text-align: center;
  padding: 20px;
}

input {
  display: block;
  margin: 10px auto;
  padding: 8px;
  font-size: 16px;
  width: 250px;
}

button {
  padding: 10px 15px;
  font-size: 18px;
  cursor: pointer;
}
</style>
