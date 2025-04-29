<template>
  <div class="container">
    <h2>FEUILLE D'INFORMATION DU PARTICIPANT </h2>
<h3>VIRAL – Intégration visionnaire pour la conception et l'apprentissage de récompenses</h3>
 
 
Investigateur principal : Valentin Cuzin-Rambaud
<br/>
Superviseur(s) [pour les étudiants] : Dr. Bruno Yun
<br/><br/>
Je suis Valentin Cuzin-Rambaud, étudiant en Master 2 IA à l'Université Claude Bernard Lyon 1, Département d'informatique.  Je vous invite à considérer la participation au projet de recherche "VIRAL" – Intégration visionnaire pour la conception et l'apprentissage de récompenses”.  Ci-dessous, vous trouverez des informations sur le projet pour vous aider à décider si vous souhaitez y participer.
<br/><br/>
La participation au projet de recherche est entièrement volontaire.  Vous pouvez vous retirer du projet à tout moment, sans avoir à donner de raison.
 
<h2>OBJECTIFS</h2>
Nous avons créé le cadre VIRAL, un pipeline pour générer et affiner les fonctions de récompense grâce à l'utilisation de LLM multimodaux. VIRAL crée de manière autonome et améliore de manière interactive les fonctions de récompense en fonction d'un environnement donné et d'une instruction ou d'une image annotée.
<br/><br/>
L'objectif de cette expérience est d'étudier la valeur de la multimodalité pour aligner le LLM avec l'objectif de l'utilisateur.

<h2>CE QUE L'ON VOUS DEMANDERA DE FAIRE</h2>
Nous aimerions que vous notiez de 1 à 5 l'alignement entre le comportement d'un agent dans une vidéo et l'instruction qu'il est censé suivre. vous pouvez ajouter un commentaire si vous avez un doute sur quelque chose. vous êtes également invité à indiquer si vous avez compris correctement l'instruction de 1 à 5. Aucune information personnelle ne sera collectée à aucun moment.

<br/><br/>
Nous prévoyons de déployer ce questionnaire pendant une période de 11 jours entre mardi 30 avril et dimanche 11 mai. Nous estimons qu'il faudra environ 1 heure pour le compléter. Le questionnaire peut être accessible à partir de n'importe quel ordinateur avec une connexion Internet.

 
<h2>RISQUES</h2>
Il ne devrait pas y avoir de risques pour vous liés à l'expérience. Vous pouvez également vous retirer de l'évaluation à tout moment.
 
<h2>GESTION ET STOCKAGE DES DONNÉES</h2>
Les données collectées seront stockées sur notre propre serveur et complètement anonymisées. Je n'enregistrerai aucune information personnelle telle que les noms ou les adresses e-mail, ni ne demanderai d'informations personnelles à aucun moment. À notre connaissance, la nature des questions ne permettra pas d'identifier rétrospectivement les utilisateurs. 

<h2>CONSENTEMENT</h2>
Si vous acceptez de participer à la recherche, votre consentement est alors déduit.

<h2>FINANCEURS</h2>
Il n'y a pas de financeurs pour cette recherche.
 
Merci de considérer la participation à cette recherche.
 
 
Si vous avez des questions sur cette recherche, veuillez me contacter :
Valentin Cuzin-Rambaud                                  	valentin.cuzin-rambaud@liris.cnrs.fr 

  </div>
  <div class="container">
    <h2>Welcome!</h2>
    <p>Enter your pseudonym</p>

    <input v-model="pseudonym" type="text" placeholder="Pseudonym" />
    <p>You need this to resume your session later.</p>
    <button @click="startRating">Start</button>
    <br/><br/><br/>
    <a href="https://github.com/VIRAL-UCBL1/VIRAL">Link to our open-source code</a>
    <br/><br/>
    <a href="https://viral-ucbl1.github.io/">Link to our website</a>

  </div>
  <div class="container">
    <h2>PARTICIPANT INFORMATION SHEET </h2>
<h3>VIRAL – Vision-grounded Integration for Reward design And Learning</h3>
 
 
Principal Investigator(s): Valentin Cuzin-Rambaud
<br/>
Supervisor(s) [for students]: Dr. Bruno Yun
<br/><br/>
I am Valentin Cuzin-Rambaud, an Master 2 AI student at the Université Claude Bernard Lyon 1, Department of Computing Science.  I would like to invite you to consider participating in the research project "VIRAL" – Vision-grounded Integration for Reward design And Learning”.  Below is some information about the project, to help you decide whether you would like to take part.
<br/><br/>
Participation in the research project is completely voluntary.  You can withdraw from the project at any time, without having to give a reason.
 
<h2>AIMS</h2>
We created the framework VIRAL, a pipeline for generating and refining reward functions through the use of multi-modal LLMs. VIRAL autonomously creates and interactively improves reward functions based on a given environment and a goal prompt or annotated image.
<br/><br/>
The aim of this experiment is to study the value of multi-modality in aligning the LLM with the user's goal.

<h2>WHAT YOU WILL BE ASKED TO DO</h2>
We would like you to score from 1 to 5 the alignment between the behavior of an agent in a video and the instruction he is asked to follow. you can add a comment if you have a doubt about anything. you are also asked if you have understood the instruction correctly from 1 to 5. No personal information will be collected at any time.

<br/><br/>
We expect to deploy this questionnaire for a period of 11 days between Tuesday 30th of april and Sunday 11th may. We estimate that it will take no longer than 1 hour to complete. The questionnaire can be accessed from any computer with an internet connection.

 
<h2>RISKS</h2>
There should not be any risks to you from the experiment. You can also withdraw from the assessment at any time.
 
<h2>DATA MANAGEMENT AND STORAGE</h2>
The data collected will be stored on our own server, and completely annonymize. I will not record any personal information such as names or email addresses, nor will I be asking any personal information at any point. To the best of our knowledge, the nature of the questions will not allow any retrospective identification of the users. 

<h2>CONSENT</h2>
If you agree to participate in the research, your consent is then deducted.

<h2>SPONSORS</h2>
There are no sponsors for this research.
 
Thank you for considering taking part in this research.
 
 
If you have any questions about this research please contact me:
Valentin Cuzin-Rambaud                                  	valentin.cuzin-rambaud@liris.cnrs.fr 

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
