import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/status': 'http://localhost:3001',  // Proxy API requests to the Express server
    },
  },
})
