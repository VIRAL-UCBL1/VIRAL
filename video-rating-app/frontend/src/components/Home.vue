<template>
  <div class="container">
    <h2>Bienvenue !</h2>
    <p>Entrez votre prénom, nom et éventuellement une seed pour reprendre votre session.</p>

    <input v-model="firstName" type="text" placeholder="Prénom" />
    <input v-model="lastName" type="text" placeholder="Nom" />
    <input v-model="userSeed" type="text" placeholder="Seed (facultatif)" />

    <button @click="startRating">Commencer</button>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const firstName = ref("");
const lastName = ref("");
const userSeed = ref("");

const generateSeed = (): string => {
  return Math.random().toString(36).substring(2, 10).toUpperCase();
};

const startRating = () => {
  if (firstName.value.trim() && lastName.value.trim()) {
    const seed = userSeed.value.trim() || generateSeed();

    const username = `${firstName.value.trim()}_${lastName.value.trim()}_${seed}`;

    // Stockage en localStorage
    localStorage.setItem("seed", seed);
    localStorage.setItem("firstName", firstName.value.trim());
    localStorage.setItem("lastName", lastName.value.trim());
    localStorage.setItem("username", username);

    // Rediriger vers la page de notation
    router.push("/rate");
  } else {
    alert("Veuillez entrer un prénom et un nom !");
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
