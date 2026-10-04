import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const api = process.env.API_PROXY || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/health": api,
      "/samples": api,
      "/results": api,
      "/result-images": api,
      "/universal-restoration": api,
      "/hard-routing": api,
      "/soft-mixture": api,
      "/face-to-sketch": api,
    },
  },
});
