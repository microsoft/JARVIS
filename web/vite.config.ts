import { fileURLToPath, URL } from "node:url";
import path  from "path";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig({
  server: { 
    host: "0.0.0.0",
    port: 9999, 
    open: true, 
    cors: true,
  },  
  plugins: [vue()],
  base: "./",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
