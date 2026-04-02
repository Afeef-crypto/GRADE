import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Bind IPv4 + IPv6 so http://127.0.0.1:5173 works (Windows/Cursor browser).
    // Do NOT pass `127.0.0.1` as a CLI arg — Vite treats it as the project root and breaks the app.
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
      // Do not proxy "/upload" alone — that is the SPA route. Only API paths:
      "/upload/key": "http://127.0.0.1:8000",
      "/upload/sheet": "http://127.0.0.1:8000",
      // GET /evaluate is the SPA page; only POST must hit FastAPI (same issue as /upload).
      "/evaluate": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        bypass(req) {
          // Vite: `return false` sends 404. Serve SPA for browser navigation to /evaluate.
          if (req.method === "GET" || req.method === "HEAD") {
            return "/index.html";
          }
        },
      },
      "/result": "http://127.0.0.1:8000",
      "/report": "http://127.0.0.1:8000",
      "/sheet": "http://127.0.0.1:8000",
    },
  },
});
