import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Vite 配置。
 *
 * 两个关键决策：
 * 1. `base: './'` —— 构建产物由后端挂在 `/ui` 下提供，用相对路径最稳；
 * 2. 开发时通过 proxy 把 `/api` 转给 FastAPI(8001)，前端代码里始终只写相对路径，
 *    生产与开发两套环境不用改代码。
 */
export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 5174,
    strictPort: true,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200
  }
})
