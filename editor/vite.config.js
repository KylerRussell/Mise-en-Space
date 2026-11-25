import { resolve } from "node:path";
import { defineConfig } from "vite";

const backend = process.env.MISE_BACKEND || "http://127.0.0.1:8765";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/files": backend,
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 4000,
    rollupOptions: { input: { main: resolve(__dirname, "index.html"), scan: resolve(__dirname, "scan.html") } },
  },
});
