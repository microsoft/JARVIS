import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import "./assets/tailwind.css";

const app = createApp(App);

app.use(router).mount("#app");
