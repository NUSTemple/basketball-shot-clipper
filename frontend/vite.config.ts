import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev-only: proxy API/media requests to the local Flask dev server so
    // the SPA can be run with `npm run dev` against a real backend without
    // CORS config - in production Flask serves both the SPA and the API
    // from the same origin (see app.py's serve_spa catch-all), so this
    // proxy has no equivalent there and isn't needed.
    proxy: {
      '/api': 'http://127.0.0.1:5050',
      // Trailing slash matters: Vite's proxy does prefix matching, and a
      // bare '/video' prefix also matches the SPA's own '/videos' and
      // '/videos/:id' client-side routes, silently sending those page
      // navigations to Flask (which 404s, since it only knows /video/...)
      // instead of letting the SPA handle them.
      '/video/': 'http://127.0.0.1:5050',
      '/thumbnail/': 'http://127.0.0.1:5050',
    },
  },
})
