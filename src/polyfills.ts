/**
 * Safari lacks ReadableStream async iterator support,
 * which phonemizer.js needs for WASM decompression.
 */
export function installStreamPolyfill(): void {
  if (
    typeof ReadableStream !== 'undefined' &&
    !(Symbol.asyncIterator in ReadableStream.prototype)
  ) {
    (ReadableStream.prototype as any)[Symbol.asyncIterator] = async function* (this: ReadableStream) {
      const reader = this.getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          yield value;
        }
      } finally {
        reader.releaseLock();
      }
    };
  }
}
