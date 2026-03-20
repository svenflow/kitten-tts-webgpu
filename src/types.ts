/** Model configuration derived from ONNX inspection. */
export interface KittenConfig {
  // Symbol table for phoneme tokenization
  symbols: string[];
  // Voice aliases
  voiceAliases: Record<string, string>;
  // Sample rate
  sampleRate: number;
}

export const DEFAULT_CONFIG: KittenConfig = {
  symbols: Array.from('$;:,.!?¡¿—…\u201c«»\u201d\u201e ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' +
    'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃ' +
    'ʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\'̩\'ᵻ'),
  voiceAliases: {
    'Bella': 'expr-voice-2-f',
    'Jasper': 'expr-voice-2-m',
    'Luna': 'expr-voice-3-f',
    'Bruno': 'expr-voice-3-m',
    'Rosie': 'expr-voice-4-f',
    'Hugo': 'expr-voice-4-m',
    'Kiki': 'expr-voice-5-f',
    'Leo': 'expr-voice-5-m',
  },
  sampleRate: 24000,
};

/** Parsed weight tensor from ONNX. */
export interface WeightTensor {
  name: string;
  shape: number[];
  dtype: 'float32' | 'float16' | 'int8' | 'uint8';
  data: ArrayBuffer;
  // For quantized weights
  scale?: Float32Array;
  zeroPoint?: Uint8Array | Int8Array;
}

/** Voice embedding data. */
export interface VoiceData {
  /** 400 style vectors per voice, each 256-dim float32 */
  embeddings: Float32Array; // [400, 256]
}

/** GPU buffer with metadata. */
export interface GpuTensor {
  buffer: GPUBuffer;
  shape: number[];
  size: number; // Total elements
}
