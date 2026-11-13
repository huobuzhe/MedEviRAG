import { defineConfig } from "vite";
export default defineConfig({
  server: {
    port: 5173,
    proxy: { "/v1": "http://127.0.0.1:8000", "/healthz": "http://127.0.0.1:8000",
             "/readyz": "http://127.0.0.1:8000" },
  },
});
