import { defineConfig } from 'vite';
import { resolve } from 'path';

// Default: demo site build. Use `npm run build:lib` for library build.
const isLib = process.env.BUILD_LIB === '1';

export default defineConfig(
  isLib
    ? {
        publicDir: false,
        build: {
          outDir: 'dist-lib',
          target: 'esnext',
          lib: {
            entry: resolve(__dirname, 'src/index.ts'),
            formats: ['es'],
            fileName: () => 'index.js',
          },
          rollupOptions: {
            // phonemizer is the only dependency — bundle it into the lib
            // (it's small and avoids users needing to install it separately)
          },
        },
      }
    : {
        base: './',
        build: {
          outDir: 'dist',
          target: 'esnext',
          rollupOptions: {
            input: {
              main: resolve(__dirname, 'index.html'),
              diag: resolve(__dirname, 'diag.html'),
            },
          },
        },
      },
);
