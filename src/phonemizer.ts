/**
 * Browser-side phonemization for Kitten TTS.
 *
 * Since espeak-ng is a native C library, we use a pre-computed lookup table
 * for common words and a simple rule-based fallback. For production, you'd
 * want espeak-ng compiled to WASM.
 *
 * For now, this module also supports receiving pre-phonemized text directly
 * (from the Python reference pipeline) for development/testing.
 */

import { DEFAULT_CONFIG } from './types.js';

/** Symbol to index mapping for the 178-symbol phoneme table. */
const symbolToIndex: Map<string, number> = new Map();
DEFAULT_CONFIG.symbols.forEach((s, i) => symbolToIndex.set(s, i));

/**
 * Convert pre-phonemized text (IPA string) to input_ids.
 *
 * The phonemizer splits tokens with regex: /\w+|[^\w\s]/g
 * then joins with spaces, looks up each char in symbol table,
 * and wraps with [0] start/end tokens.
 */
export function phonemesToInputIds(phonemes: string): number[] {
  // Split into words and punctuation, join with spaces
  const tokens = phonemes.match(/\w+|[^\w\s]/g) || [];
  const joined = tokens.join(' ');

  // Map characters to indices, skip unknown
  const ids: number[] = [0]; // Start token
  for (const char of joined) {
    const idx = symbolToIndex.get(char);
    if (idx !== undefined) {
      ids.push(idx);
    }
  }
  ids.push(0); // End token

  return ids;
}

/**
 * Simple English text-to-phoneme converter for demo purposes.
 * Uses a lookup table for common words. Falls back to the text itself
 * (which won't produce correct phonemes, but keeps the pipeline working).
 *
 * For correct results, use the Python pipeline or espeak-ng WASM.
 */
const WORD_PHONEMES: Record<string, string> = {
  'hello': 'həlˈoʊ',
  'this': 'ðɪs',
  'is': 'ɪz',
  'a': 'ɐ',
  'test': 'tˈɛst',
  'the': 'ðə',
  'and': 'ænd',
  'of': 'ʌv',
  'to': 'tə',
  'in': 'ɪn',
  'for': 'fɔːɹ',
  'it': 'ɪt',
  'you': 'juː',
  'that': 'ðæt',
  'was': 'wʌz',
  'on': 'ɑːn',
  'are': 'ɑːɹ',
  'with': 'wɪð',
  'they': 'ðeɪ',
  'be': 'biː',
  'at': 'æt',
  'have': 'hæv',
  'from': 'fɹʌm',
  'or': 'ɔːɹ',
  'had': 'hæd',
  'but': 'bʌt',
  'not': 'nɑːt',
  'what': 'wʌt',
  'all': 'ɔːl',
  'were': 'wɜːɹ',
  'we': 'wiː',
  'when': 'wɛn',
  'can': 'kæn',
  'there': 'ðɛɹ',
  'an': 'æn',
  'your': 'jʊɹ',
  'which': 'wɪtʃ',
  'do': 'duː',
  'how': 'haʊ',
  'if': 'ɪf',
  'will': 'wɪl',
  'up': 'ʌp',
  'about': 'ɐbˈaʊt',
  'out': 'aʊt',
  'so': 'soʊ',
  'my': 'maɪ',
  'one': 'wʌn',
  'i': 'aɪ',
  'world': 'wˈɜːld',
  'welcome': 'wˈɛlkʌm',
  'good': 'ɡʊd',
  'morning': 'mˈɔːɹnɪŋ',
  'thank': 'θæŋk',
  'please': 'pliːz',
  'yes': 'jˈɛs',
  'no': 'noʊ',
  'okay': 'oʊkˈeɪ',
};

export function textToPhonemes(text: string): string {
  const words = text.toLowerCase().match(/\w+|[^\w\s]/g) || [];
  return words.map(w => WORD_PHONEMES[w] || w).join(' ');
}

export function textToInputIds(text: string): number[] {
  return phonemesToInputIds(textToPhonemes(text));
}
