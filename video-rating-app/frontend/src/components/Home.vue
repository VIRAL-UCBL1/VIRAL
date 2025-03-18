<template>
    <div class="container">
      <h2>Bienvenue !</h2> <!-- Welcome message in French -->
      <p>Entrez votre prénom et nom pour commencer à évaluer les vidéos.</p> <!-- Instruction to enter first and last name in French -->
  
      <!-- Input fields for first name and last name with two-way data binding -->
      <input v-model="firstName" type="text" placeholder="Prénom" />  <!-- First name input -->
      <input v-model="lastName" type="text" placeholder="Nom" />  <!-- Last name input -->
      <!-- Button to start rating videos -->
      <button @click="startRating">Commencer</button> <!-- Start button to begin rating -->
    </div>
</template>
  
<script setup lang="ts">
import { ref } from "vue"; // Import Vue's ref function to create reactive references
import { useRouter } from "vue-router"; // Import useRouter for navigation

const router = useRouter();  // Initialize the router
const firstName = ref("");  // Reactive reference for the first name input
const lastName = ref("");  // Reactive reference for the last name input

// Function to start the rating process
const startRating = () => {
  // Check if both first name and last name are filled out
  if (firstName.value.trim() && lastName.value.trim()) {
    // Create a username by combining first name and last name
    const username = `${firstName.value}_${lastName.value}`;
    // Store the username in local storage
    localStorage.setItem("username", username);
    // Redirect to the rating page
    router.push("/rate"); // Redirection to the rating page
  } else {
    // If the fields are empty, show an alert message
    alert("Veuillez entrer un prénom et un nom !"); // Alert in French
  }
};
</script>
  
<style scoped>
/* Center the container and add padding */
.container {
  text-align: center;
  padding: 20px;
}

/* Style for input fields */
input {
  display: block;
  margin: 10px auto;
  padding: 8px;
  font-size: 16px;
}

/* Style for the button */
button {
  padding: 10px 15px;
  font-size: 18px;
  cursor: pointer;  /* Change the cursor to pointer when hovering over the button */
}
</style>
