import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_ACTIONS ? "/rallylens/" : "/",
  worker: { format: "es" },
  define: { __APP_VERSION__: JSON.stringify(process.env.npm_package_version ?? "dev") },
});
