<template>
    <div class="container">
      <h2>Bienvenue !</h2>
      <p>Entrez votre prénom et nom pour commencer à évaluer les vidéos.</p>
  
      <input v-model="firstName" type="text" placeholder="Prénom" />
      <input v-model="lastName" type="text" placeholder="Nom" />
      <button @click="startRating">Commencer</button>
    </div>
  </template>
  
  <script setup lang="ts">
  import { ref } from "vue";
import { useRouter } from "vue-router";
  
  const router = useRouter();
  const firstName = ref("");
  const lastName = ref("");
  
  const startRating = () => {
    if (firstName.value.trim() && lastName.value.trim()) {
      const username = `${firstName.value}_${lastName.value}`;
      localStorage.setItem("username", username);
      router.push("/rate"); // Redirection vers la notation
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
  }
  button {
    padding: 10px 15px;
    font-size: 18px;
    cursor: pointer;
  }
  </style>
  