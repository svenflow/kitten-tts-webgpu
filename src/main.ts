/**
 * Main entry point for Kitten TTS WebGPU demo.
 */

import { KittenTTSEngine } from './engine.js';
import { textToInputIds } from './phonemizer.js';

const logEl = document.getElementById('log')!;
const btnEl = document.getElementById('generate') as HTMLButtonElement;
const textEl = document.getElementById('text') as HTMLTextAreaElement;
const voiceEl = document.getElementById('voice') as HTMLSelectElement;
const speedEl = document.getElementById('speed') as HTMLInputElement;
const speedValEl = document.getElementById('speed-val')!;
const audioEl = document.getElementById('audio') as HTMLAudioElement;

function log(msg: string, type: 'info' | 'error' | 'success' = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
  console.log(`[${type}] ${msg}`);
}

// Speed slider
speedEl.addEventListener('input', () => {
  speedValEl.textContent = `${speedEl.value}×`;
});

async function main() {
  log('Initializing WebGPU...');

  if (!navigator.gpu) {
    log('WebGPU not available in this browser!', 'error');
    return;
  }

  const engine = new KittenTTSEngine();
  engine.profile = true; // Enable timing instrumentation

  try {
    await engine.init();
    log('WebGPU device ready', 'success');
  } catch (e) {
    log(`WebGPU init failed: ${e}`, 'error');
    return;
  }

  // Load model — use local path for dev, HuggingFace CDN for production (GitHub Pages)
  log('Loading model weights (74.6 MB)...');
  const HF_BASE = 'https://huggingface.co/hexgrad/Kitten-TTS/resolve/main';
  const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  const modelUrl = isLocal
    ? '/models/kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx'
    : `${HF_BASE}/kitten_tts_mini_v0_8.onnx`;
  const voicesUrl = isLocal
    ? '/models/kitten-tts-mini-0.8/voices.npz'
    : `${HF_BASE}/voices.npz`;

  try {
    const start = performance.now();
    await engine.loadModel(modelUrl, voicesUrl);
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    log(`Model loaded in ${elapsed}s`, 'success');
  } catch (e) {
    log(`Model load failed: ${e}`, 'error');
    return;
  }

  btnEl.disabled = false;
  btnEl.textContent = 'Generate Speech';

  btnEl.addEventListener('click', async () => {
    btnEl.disabled = true;
    btnEl.textContent = 'Generating...';

    const text = textEl.value.trim();
    const voice = voiceEl.value;
    const speed = parseFloat(speedEl.value);

    if (!text) {
      log('Please enter some text!', 'error');
      btnEl.disabled = false;
      btnEl.textContent = 'Generate Speech';
      return;
    }

    log(`Generating: "${text}" (voice=${voice}, speed=${speed}×)`);

    try {
      // Phonemize
      const inputIds = textToInputIds(text);
      log(`Phonemized: ${inputIds.length} tokens`);

      // Run inference
      const start = performance.now();
      const { waveform, duration } = await engine.generate(inputIds, voice, speed);
      const elapsed = ((performance.now() - start) / 1000).toFixed(2);

      log(`Generated ${waveform.length} samples (${(waveform.length / 24000).toFixed(2)}s) in ${elapsed}s`, 'success');

      // Convert to WAV and play
      const wavBlob = float32ToWav(waveform, 24000);
      const url = URL.createObjectURL(wavBlob);
      audioEl.src = url;
      audioEl.style.display = 'block';
      audioEl.play();

    } catch (e) {
      log(`Generation failed: ${e}`, 'error');
    }

    btnEl.disabled = false;
    btnEl.textContent = 'Generate Speech';
  });
}

/** Convert Float32Array audio to WAV blob. */
function float32ToWav(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  // WAV header
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert float32 [-1,1] to int16
  let offset = headerSize;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

main();
