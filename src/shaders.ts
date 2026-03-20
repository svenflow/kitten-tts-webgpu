/**
 * WGSL compute shaders for Kitten TTS WebGPU inference.
 *
 * The model has these major shader requirements:
 * 1. Embedding lookup (word + position)
 * 2. Layer Normalization
 * 3. Matrix multiplication (with quantized weight dequant)
 * 4. Multi-head attention (ALBERT encoder)
 * 5. Conv1d (text encoder CNN, predictor, decoder, generator)
 * 6. LSTM (text encoder, predictor, shared)
 * 7. Instance Normalization (decoder)
 * 8. Adaptive Instance Normalization (AdaIN - style conditioning)
 * 9. ConvTranspose1d (HiFi-GAN upsampling)
 * 10. LeakyReLU, GELU, Tanh, Sigmoid activations
 * 11. Residual connections
 */

// ── Embedding Lookup ─────────────────────────────────────────────────────────

export const embeddingShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> input_ids: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  seq_len: u32,
  embed_dim: u32,
  vocab_size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let seq_idx = idx / params.embed_dim;
  let dim_idx = idx % params.embed_dim;

  if (seq_idx >= params.seq_len) { return; }

  let token_id = input_ids[seq_idx];
  let embed_offset = u32(token_id) * params.embed_dim + dim_idx;
  output[idx] = embeddings[embed_offset];
}
`;

// ── Layer Normalization ──────────────────────────────────────────────────────

export const layerNormShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
  batch_size: u32,
  hidden_size: u32,
  eps: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch_idx = gid.x;
  if (batch_idx >= params.batch_size) { return; }

  let offset = batch_idx * params.hidden_size;

  // Compute mean
  var sum = 0.0;
  for (var i = 0u; i < params.hidden_size; i++) {
    sum += input[offset + i];
  }
  let mean = sum / f32(params.hidden_size);

  // Compute variance
  var var_sum = 0.0;
  for (var i = 0u; i < params.hidden_size; i++) {
    let diff = input[offset + i] - mean;
    var_sum += diff * diff;
  }
  let variance = var_sum / f32(params.hidden_size);
  let inv_std = 1.0 / sqrt(variance + params.eps);

  // Normalize
  for (var i = 0u; i < params.hidden_size; i++) {
    output[offset + i] = (input[offset + i] - mean) * inv_std * gamma[i] + beta[i];
  }
}
`;

// ── Matrix Multiply (general) ────────────────────────────────────────────────

export const matmulShader = /* wgsl */ `
// Tiled matmul with shared memory. TILE=16, each workgroup computes a 16×16 output tile.
// Reduces global memory reads by factor of TILE compared to naive approach.

const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
  M: u32,  // rows of A / output
  K: u32,  // cols of A / rows of B
  N: u32,  // cols of B / output
  use_bias: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 256>;  // 16×16
var<workgroup> tileB: array<f32, 256>;  // 16×16

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.x;
  let col = gid.y;
  let lr = lid.x;
  let lc = lid.y;

  var sum = 0.0;
  let numTiles = (params.K + TILE - 1u) / TILE;

  for (var t = 0u; t < numTiles; t++) {
    // Load tile of A: rows [row_base..+16], cols [t*16..+16]
    let aCol = t * TILE + lc;
    if (row < params.M && aCol < params.K) {
      tileA[lr * TILE + lc] = A[row * params.K + aCol];
    } else {
      tileA[lr * TILE + lc] = 0.0;
    }

    // Load tile of B: rows [t*16..+16], cols [col_base..+16]
    let bRow = t * TILE + lr;
    if (bRow < params.K && col < params.N) {
      tileB[lr * TILE + lc] = B[bRow * params.N + col];
    } else {
      tileB[lr * TILE + lc] = 0.0;
    }

    workgroupBarrier();

    // Accumulate dot product from shared memory
    for (var k = 0u; k < TILE; k++) {
      sum += tileA[lr * TILE + k] * tileB[k * TILE + lc];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    if (params.use_bias != 0u) {
      sum += bias[col];
    }
    output[row * params.N + col] = sum;
  }
}
`;

// ── Conv1d ───────────────────────────────────────────────────────────────────

export const conv1dShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;     // [C_in, L]
@group(0) @binding(1) var<storage, read> weight: array<f32>;    // [C_out, C_in, K]
@group(0) @binding(2) var<storage, read> bias: array<f32>;      // [C_out]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [C_out, L_out]

struct Params {
  in_channels: u32,
  out_channels: u32,
  kernel_size: u32,
  input_length: u32,
  output_length: u32,
  padding: u32,
  stride: u32,
  dilation: u32,
  use_bias: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let out_ch = idx / params.output_length;
  let out_pos = idx % params.output_length;

  if (out_ch >= params.out_channels) { return; }

  var sum = 0.0;
  for (var ic = 0u; ic < params.in_channels; ic++) {
    for (var k = 0u; k < params.kernel_size; k++) {
      let in_pos_raw = i32(out_pos * params.stride) + i32(k * params.dilation) - i32(params.padding);
      if (in_pos_raw >= 0 && u32(in_pos_raw) < params.input_length) {
        let w_idx = out_ch * params.in_channels * params.kernel_size + ic * params.kernel_size + k;
        let in_idx = ic * params.input_length + u32(in_pos_raw);
        sum += input[in_idx] * weight[w_idx];
      }
    }
  }

  if (params.use_bias != 0u) {
    sum += bias[out_ch];
  }

  output[out_ch * params.output_length + out_pos] = sum;
}
`;

// ── Instance Normalization ───────────────────────────────────────────────────

export const instanceNormShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;     // [C, L]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [C, L]

struct Params {
  channels: u32,
  length: u32,
  eps: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= params.channels) { return; }

  let offset = ch * params.length;

  // Compute mean
  var sum = 0.0;
  for (var i = 0u; i < params.length; i++) {
    sum += input[offset + i];
  }
  let mean = sum / f32(params.length);

  // Compute variance
  var var_sum = 0.0;
  for (var i = 0u; i < params.length; i++) {
    let diff = input[offset + i] - mean;
    var_sum += diff * diff;
  }
  let variance = var_sum / f32(params.length);
  let inv_std = 1.0 / sqrt(variance + params.eps);

  // Normalize (no scale/bias for instance norm in this model - AdaIN handles that)
  for (var i = 0u; i < params.length; i++) {
    output[offset + i] = (input[offset + i] - mean) * inv_std;
  }
}
`;

// ── Adaptive Instance Normalization (AdaIN) ──────────────────────────────────

export const adainShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> normed: array<f32>;    // [C, L] - instance-normed input
@group(0) @binding(1) var<storage, read> style_fc: array<f32>;  // [2*C] - first C = scale, second C = bias
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [C, L]

struct Params {
  channels: u32,
  length: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let ch = idx / params.length;
  let pos = idx % params.length;

  if (ch >= params.channels) { return; }

  // AdaIN: (1 + gamma) * normed + beta — the +1 offset is universal across all AdaIN blocks
  // style_fc layout: [scale_0..scale_{C-1}, bias_0..bias_{C-1}]
  let scale = style_fc[ch];
  let bias = style_fc[params.channels + ch];
  output[idx] = normed[idx] * (scale + 1.0) + bias;
}
`;

// ── AdaIN row-major: normed[rows, C] + style_fc[2*C] → output[rows, C] ──

export const adainRowMajorShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> normed: array<f32>;    // [rows, C]
@group(0) @binding(1) var<storage, read> style_fc: array<f32>;  // [2*C]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [rows, C]

struct Params {
  channels: u32,
  total: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total) { return; }

  // Row-major: channel = idx % channels
  let ch = idx % params.channels;
  let scale = style_fc[ch];
  let bias = style_fc[params.channels + ch];
  output[idx] = normed[idx] * (scale + 1.0) + bias;
}
`;

// ── Snake Activation (HiFi-GAN) ──────────────────────────────────────────────
// Snake(x, alpha) = x + (1/alpha) * sin²(alpha * x)
// alpha is per-channel: [1, C, 1], applied across the temporal dimension

export const snakeShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;     // [C, L]
@group(0) @binding(1) var<storage, read> alpha: array<f32>;     // [C] (flattened from [1, C, 1])
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [C, L]

struct Params {
  channels: u32,
  length: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let ch = idx / params.length;
  let pos = idx % params.length;

  if (ch >= params.channels) { return; }

  let x = input[idx];
  let a = alpha[ch];
  let sin_ax = sin(a * x);
  // Snake: x + (1/a) * sin²(a * x)
  output[idx] = x + sin_ax * sin_ax / a;
}
`;

// ── LSTM: See bidirectional LSTM shader at bottom of file ───────────────────

// ── Activations ──────────────────────────────────────────────────────────────

export const leakyReluShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  size: u32,
  alpha: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let x = input[idx];
  output[idx] = select(params.alpha * x, x, x >= 0.0);
}
`;

export const geluShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let x = input[idx];
  // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  // Clamp tanh arg to prevent exp(2x) overflow in f32 (exp overflows at ~88.72)
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = clamp(c * (x + 0.044715 * x * x * x), -44.0, 44.0);
  output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
`;

export const tanhShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params { size: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = tanh(input[idx]);
}
`;

export const sigmoidShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params { size: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}
`;

// ── ConvTranspose1d (for HiFi-GAN upsampling) ───────────────────────────────

export const convTranspose1dShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;     // [C_in, L_in]
@group(0) @binding(1) var<storage, read> weight: array<f32>;    // [C_in, C_out, K]
@group(0) @binding(2) var<storage, read> bias: array<f32>;      // [C_out]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [C_out, L_out]

struct Params {
  in_channels: u32,
  out_channels: u32,
  kernel_size: u32,
  input_length: u32,
  output_length: u32,
  stride: u32,
  padding: u32,
  use_bias: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let out_ch = idx / params.output_length;
  let out_pos = idx % params.output_length;

  if (out_ch >= params.out_channels) { return; }

  var sum = 0.0;
  for (var ic = 0u; ic < params.in_channels; ic++) {
    for (var k = 0u; k < params.kernel_size; k++) {
      // ConvTranspose: output[out_pos] += input[in_pos] * weight[ic, out_ch, k]
      // where out_pos = in_pos * stride + k - padding
      // so in_pos = (out_pos + padding - k) / stride
      let numerator = i32(out_pos) + i32(params.padding) - i32(k);
      if (numerator >= 0 && u32(numerator) % params.stride == 0u) {
        let in_pos = u32(numerator) / params.stride;
        if (in_pos < params.input_length) {
          let w_idx = ic * params.out_channels * params.kernel_size + out_ch * params.kernel_size + k;
          let in_idx = ic * params.input_length + in_pos;
          sum += input[in_idx] * weight[w_idx];
        }
      }
    }
  }

  if (params.use_bias != 0u) {
    sum += bias[out_ch];
  }

  output[out_ch * params.output_length + out_pos] = sum;
}
`;

// ── Depthwise ConvTranspose1d (for pool layers: groups=channels) ────────────

export const depthwiseConvTranspose1dShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;     // [channels, L_in]
@group(0) @binding(1) var<storage, read> weight: array<f32>;    // [channels, 1, K] = [channels * K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [channels, L_out]

struct Params {
  channels: u32,
  kernel_size: u32,
  input_length: u32,
  output_length: u32,
  stride: u32,
  padding: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let ch = idx / params.output_length;
  let out_pos = idx % params.output_length;

  if (ch >= params.channels) { return; }

  var sum = 0.0;
  for (var k = 0u; k < params.kernel_size; k++) {
    let numerator = i32(out_pos) + i32(params.padding) - i32(k);
    if (numerator >= 0 && u32(numerator) % params.stride == 0u) {
      let in_pos = u32(numerator) / params.stride;
      if (in_pos < params.input_length) {
        let w_idx = ch * params.kernel_size + k;
        let in_idx = ch * params.input_length + in_pos;
        sum += input[in_idx] * weight[w_idx];
      }
    }
  }

  output[ch * params.output_length + out_pos] = sum;
}
`;

// ── Resize 1D (nearest-neighbor 2x upsample) ───────────────────────────────

export const resize1dShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;      // [channels, L_in]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [channels, L_out]

struct Params {
  channels: u32,
  input_length: u32,
  output_length: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let ch = idx / params.output_length;
  let out_pos = idx % params.output_length;

  if (ch >= params.channels) { return; }

  // Nearest neighbor: map output position to input position
  let in_pos = out_pos * params.input_length / params.output_length;
  output[ch * params.output_length + out_pos] = input[ch * params.input_length + in_pos];
}
`;

// ── Softmax ──────────────────────────────────────────────────────────────────

export const softmaxShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  batch_size: u32,
  dim_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch_idx = gid.x;
  if (batch_idx >= params.batch_size) { return; }

  let offset = batch_idx * params.dim_size;

  // Find max for numerical stability
  var max_val = input[offset];
  for (var i = 1u; i < params.dim_size; i++) {
    max_val = max(max_val, input[offset + i]);
  }

  // Compute exp and sum
  var exp_sum = 0.0;
  for (var i = 0u; i < params.dim_size; i++) {
    let e = exp(input[offset + i] - max_val);
    output[offset + i] = e;
    exp_sum += e;
  }

  // Normalize
  for (var i = 0u; i < params.dim_size; i++) {
    output[offset + i] /= exp_sum;
  }
}
`;

// ── Multi-Head Attention (fused QK^T softmax V) ─────────────────────────────
// One workgroup per (batch_head, query_pos) pair.
// Each thread handles one key position for the dot product, then we do
// workgroup-level reduction for softmax, then each thread accumulates V.

export const mhaShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> Q: array<f32>;  // [seq_len, num_heads, head_dim]
@group(0) @binding(1) var<storage, read> K: array<f32>;  // [seq_len, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> V: array<f32>;  // [seq_len, num_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [seq_len, num_heads, head_dim]

struct Params {
  seq_len: u32,
  num_heads: u32,
  head_dim: u32,
  scale: f32,  // 1/sqrt(head_dim)
}
@group(0) @binding(4) var<uniform> params: Params;

// Workgroup: one per (head, query_pos). Threads iterate over key positions.
// We use a simple approach: each thread computes one output element (head_dim index).

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // gid.x = dim_idx within head, gid.y = head_idx * seq_len + query_pos
  let dim_idx = gid.x;
  let head_query = gid.y;
  let head_idx = head_query / params.seq_len;
  let q_pos = head_query % params.seq_len;

  if (dim_idx >= params.head_dim || head_idx >= params.num_heads) { return; }

  let hd = params.head_dim;
  let nh = params.num_heads;
  let sl = params.seq_len;

  // Q vector for this (q_pos, head): Q[q_pos * nh * hd + head_idx * hd + ...]
  let q_base = q_pos * nh * hd + head_idx * hd;

  // Compute attention scores: dot(Q[q_pos, head], K[k_pos, head]) for all k_pos
  // Then softmax and weighted sum of V
  // Since we can't do cross-thread softmax easily, each thread computes full attention
  // for one output dimension. This is O(seq_len * head_dim) per thread but simple.

  // Step 1: Compute all attention scores (each thread does this redundantly)
  // For short sequences (< 512) this is fine
  var max_score = -1e10;
  for (var k = 0u; k < sl; k++) {
    let k_base = k * nh * hd + head_idx * hd;
    var score = 0.0;
    for (var d = 0u; d < hd; d++) {
      score += Q[q_base + d] * K[k_base + d];
    }
    score *= params.scale;
    max_score = max(max_score, score);
  }

  // Step 2: Softmax
  var exp_sum = 0.0;
  var weighted_val = 0.0;
  for (var k = 0u; k < sl; k++) {
    let k_base = k * nh * hd + head_idx * hd;
    var score = 0.0;
    for (var d = 0u; d < hd; d++) {
      score += Q[q_base + d] * K[k_base + d];
    }
    score *= params.scale;
    let w = exp(score - max_score);
    exp_sum += w;

    // Accumulate V[k_pos, head, dim_idx] weighted by attention
    let v_base = k * nh * hd + head_idx * hd;
    weighted_val += w * V[v_base + dim_idx];
  }

  let out_idx = q_pos * nh * hd + head_idx * hd + dim_idx;
  output[out_idx] = weighted_val / exp_sum;
}
`;

// ── Matmul + GELU fused ─────────────────────────────────────────────────────

export const matmulGeluShader = /* wgsl */ `
// Tiled matmul + GELU with shared memory.

const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
  M: u32,
  K: u32,
  N: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.x;
  let col = gid.y;
  let lr = lid.x;
  let lc = lid.y;

  var sum = 0.0;
  let numTiles = (params.K + TILE - 1u) / TILE;

  for (var t = 0u; t < numTiles; t++) {
    let aCol = t * TILE + lc;
    if (row < params.M && aCol < params.K) {
      tileA[lr * TILE + lc] = A[row * params.K + aCol];
    } else {
      tileA[lr * TILE + lc] = 0.0;
    }

    let bRow = t * TILE + lr;
    if (bRow < params.K && col < params.N) {
      tileB[lr * TILE + lc] = B[bRow * params.N + col];
    } else {
      tileB[lr * TILE + lc] = 0.0;
    }

    workgroupBarrier();

    for (var k = 0u; k < TILE; k++) {
      sum += tileA[lr * TILE + k] * tileB[k * TILE + lc];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    sum += bias[col];
    // GELU activation (clamp tanh arg to prevent f32 exp overflow)
    let c = 0.7978845608;
    let x = sum;
    let inner = clamp(c * (x + 0.044715 * x * x * x), -44.0, 44.0);
    output[row * params.N + col] = 0.5 * x * (1.0 + tanh(inner));
  }
}
`;

// ── Reshape for MHA: [seq, hidden] → [seq, heads, head_dim] ────────────────
// This is a no-op in memory (same layout) but we need a shader to split
// the linear projection output into Q, K, V with proper head interleaving

// ── Element-wise Add ─────────────────────────────────────────────────────────

export const addShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params { size: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = a[idx] + b[idx];
}
`;

// ── Scale (multiply by constant) ────────────────────────────────────────────

export const scaleShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  size: u32,
  _pad1: u32,
  scale: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input[idx] * params.scale;
}
`;

// ── Concat channels-first: concatenate N channel-first tensors along channel dim ──

export const concatChannelsShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;      // [C_a, L]
@group(0) @binding(1) var<storage, read> b: array<f32>;      // [C_b, L]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [C_a + C_b, L]

struct Params {
  channels_a: u32,
  channels_b: u32,
  length: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = (params.channels_a + params.channels_b) * params.length;
  if (idx >= total) { return; }

  let ch = idx / params.length;
  let pos = idx % params.length;

  if (ch < params.channels_a) {
    output[idx] = a[ch * params.length + pos];
  } else {
    output[idx] = b[(ch - params.channels_a) * params.length + pos];
  }
}
`;

// ── Concat row-major with broadcast: A[rows, colsA] + B[colsB] → [rows, colsA+colsB] ──

export const concatBroadcastShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;      // [rows, cols_a]
@group(0) @binding(1) var<storage, read> b: array<f32>;      // [cols_b] — broadcast to every row
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [rows, cols_a + cols_b]

struct Params {
  rows: u32,
  cols_a: u32,
  cols_b: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_cols = params.cols_a + params.cols_b;
  let total = params.rows * total_cols;
  if (idx >= total) { return; }

  let row = idx / total_cols;
  let col = idx % total_cols;

  if (col < params.cols_a) {
    output[idx] = a[row * params.cols_a + col];
  } else {
    output[idx] = b[col - params.cols_a];
  }
}
`;

// ── Reflection Pad 1D: add samples at start/end via reflection ──────────────

export const reflectionPad1dShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;      // [channels, L_in]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [channels, L_out]

struct Params {
  channels: u32,
  input_length: u32,
  pad_left: u32,
  pad_right: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let out_length = params.input_length + params.pad_left + params.pad_right;
  let ch = idx / out_length;
  let out_pos = idx % out_length;

  if (ch >= params.channels) { return; }

  var in_pos: u32;
  if (out_pos < params.pad_left) {
    // Reflected left: position 0 -> pad_left, position 1 -> pad_left-1, etc.
    in_pos = params.pad_left - out_pos;
  } else if (out_pos >= params.pad_left + params.input_length) {
    // Reflected right
    let overshoot = out_pos - params.pad_left - params.input_length;
    in_pos = params.input_length - 2u - overshoot;
  } else {
    in_pos = out_pos - params.pad_left;
  }

  output[ch * out_length + out_pos] = input[ch * params.input_length + in_pos];
}
`;

// ── Alpha-weighted residual add (for HiFi-GAN resblocks) ────────────────────

export const alphaResidualShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> current: array<f32>;    // conv2 output
@group(0) @binding(1) var<storage, read> residual: array<f32>;   // residual from previous iteration
@group(0) @binding(2) var<storage, read> alpha: array<f32>;      // [1, channels, 1] per-channel alpha
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
  channels: u32,
  length: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let ch = idx / params.length;
  if (ch >= params.channels) { return; }

  // output = current + alpha[ch] * residual
  output[idx] = current[idx] + alpha[ch] * residual[idx];
}
`;

// ── Transpose 2D ────────────────────────────────────────────────────────────
// Transposes [rows, cols] → [cols, rows]

export const transposeShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params { rows: u32, cols: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.rows * params.cols;
  if (idx >= total) { return; }

  let row = idx / params.cols;
  let col = idx % params.cols;
  output[col * params.rows + row] = input[idx];
}
`;

// ── Bidirectional LSTM ──────────────────────────────────────────────────────
// Runs a full bidirectional LSTM over the sequence.
// Each workgroup thread handles one hidden unit.
// Uses sequential processing per time step (inherent LSTM constraint).
//
// Weights stored in DynamicQuantizeLSTM / MatMulInteger layout:
//   W = [num_dir, input_size, 4*hidden_size]   (for x @ W matmul)
//   R = [num_dir, hidden_size, 4*hidden_size]   (for h @ R matmul)
//   B = [num_dir, 8*hidden_size]               (bias: Wb_i,Wb_o,Wb_f,Wb_c,Rb_i,Rb_o,Rb_f,Rb_c)
//
// ONNX LSTM gate order: i, o, f, c (input, output, forget, cell)
// Output: [seq_len, num_dir, hidden_size]

export const lstmShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;   // [seq_len, input_size]
@group(0) @binding(1) var<storage, read> W: array<f32>;       // [num_dir, input_size, 4*hidden]
@group(0) @binding(2) var<storage, read> R: array<f32>;       // [num_dir, hidden, 4*hidden]
@group(0) @binding(3) var<storage, read> bias: array<f32>;    // [num_dir, 8*hidden]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [seq_len, num_dir, hidden]

struct Params {
  seq_len: u32,
  input_size: u32,
  hidden_size: u32,
  num_directions: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h_idx = gid.x; // which hidden unit
  let dir = gid.y; // 0=forward, 1=backward
  // NOTE: no early return — all threads in workgroup must reach storageBarrier()
  let is_valid = h_idx < params.hidden_size && dir < params.num_directions;

  let H = params.hidden_size;
  let H4 = H * 4u;
  let IS = params.input_size;
  let SL = params.seq_len;

  // Use safe indices for inactive threads (they won't write)
  let safe_h = select(0u, h_idx, is_valid);
  let safe_dir = select(0u, dir, is_valid);

  // Gate offsets within 4*hidden: i=0, o=1, f=2, c=3 (ONNX order)
  let gate_i = safe_h;
  let gate_o = H + safe_h;
  let gate_f = 2u * H + safe_h;
  let gate_c = 3u * H + safe_h;

  // Bias offsets: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
  let bias_base = safe_dir * 8u * H;
  var b_wi = 0.0; var b_wo = 0.0; var b_wf = 0.0; var b_wc = 0.0;
  var b_ri = 0.0; var b_ro = 0.0; var b_rf = 0.0; var b_rc = 0.0;
  if (is_valid) {
    b_wi = bias[bias_base + safe_h];
    b_wo = bias[bias_base + H + safe_h];
    b_wf = bias[bias_base + 2u * H + safe_h];
    b_wc = bias[bias_base + 3u * H + safe_h];
    b_ri = bias[bias_base + 4u * H + safe_h];
    b_ro = bias[bias_base + 5u * H + safe_h];
    b_rf = bias[bias_base + 6u * H + safe_h];
    b_rc = bias[bias_base + 7u * H + safe_h];
  }

  var h_val = 0.0; // hidden state for this unit
  var c_val = 0.0; // cell state for this unit

  // Weight base offsets for this direction
  // W: [num_dir, IS, 4H] — flat stride: dir * IS * H4
  // R: [num_dir, H, 4H]  — flat stride: dir * H * H4
  let w_base = safe_dir * IS * H4;
  let r_base = safe_dir * H * H4;

  for (var step = 0u; step < SL; step++) {
    if (is_valid) {
      // Forward: t=step, Backward: t=SL-1-step
      let t = select(SL - 1u - step, step, safe_dir == 0u);

      // Compute gates from input: sum over input_size
      var gi = b_wi + b_ri;
      var go = b_wo + b_ro;
      var gf = b_wf + b_rf;
      var gc = b_wc + b_rc;

      // Input contribution: W[dir, j, gate*H+h_idx] — layout [IS, 4H]
      // x[j] * W[w_base + j * H4 + gate_offset]
      for (var j = 0u; j < IS; j++) {
        let x_val = input[t * IS + j];
        let w_off = w_base + j * H4;
        gi += x_val * W[w_off + gate_i];
        go += x_val * W[w_off + gate_o];
        gf += x_val * W[w_off + gate_f];
        gc += x_val * W[w_off + gate_c];
      }

      // Recurrence contribution: R[dir, j, gate*H+h_idx] — layout [H, 4H]
      // h_prev[j] * R[r_base + j * H4 + gate_offset]
      if (step > 0u) {
        let prev_t = select(SL - step, step - 1u, safe_dir == 0u);
        let prev_base = prev_t * params.num_directions * H + safe_dir * H;
        for (var j = 0u; j < H; j++) {
          let h_prev = output[prev_base + j];
          let r_off = r_base + j * H4;
          gi += h_prev * R[r_off + gate_i];
          go += h_prev * R[r_off + gate_o];
          gf += h_prev * R[r_off + gate_f];
          gc += h_prev * R[r_off + gate_c];
        }
      }

      // Apply activations
      // Clamp sigmoid inputs to avoid exp overflow (exp(88.72) > f32 max)
      let i_gate = 1.0 / (1.0 + exp(-clamp(gi, -44.0, 44.0))); // sigmoid
      let o_gate = 1.0 / (1.0 + exp(-clamp(go, -44.0, 44.0)));
      let f_gate = 1.0 / (1.0 + exp(-clamp(gf, -44.0, 44.0)));
      // Clamp tanh inputs: tanh uses exp(2x), so |x| > 44 → exp(88) → Inf → NaN
      let c_gate = tanh(clamp(gc, -44.0, 44.0));

      c_val = f_gate * c_val + i_gate * c_gate;
      h_val = o_gate * tanh(clamp(c_val, -44.0, 44.0));

      // Write output: [t, dir, h_idx] → flat: t * num_dir * H + dir * H + h_idx
      output[t * params.num_directions * H + safe_dir * H + safe_h] = h_val;
    }

    // Barrier: all threads (active and inactive) must reach this point
    // storageBarrier() ensures visibility of storage buffer writes across threads in the workgroup
    storageBarrier();
  }
}
`;

// Length expansion: [seqLen, D] → [totalFrames, D] using duration cumsum
// Each thread: find source token via binary search on cumsum, copy D-dim vector
export const expandRowMajorShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;      // [seq_len, dim]
@group(0) @binding(1) var<storage, read> cumsum: array<u32>;     // [seq_len] prefix sum of durations
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [total_frames, dim]

struct Params {
  seq_len: u32,
  dim: u32,
  total_frames: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.total_frames * params.dim;
  if (idx >= total) { return; }

  let frame = idx / params.dim;
  let d = idx % params.dim;

  // Binary search: find token i where cumsum[i-1] <= frame < cumsum[i]
  var lo: u32 = 0u;
  var hi: u32 = params.seq_len;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (cumsum[mid] <= frame) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  let token = lo;

  output[idx] = input[token * params.dim + d];
}
`;

// Length expansion with transpose: [seqLen, D] row-major → [D, totalFrames] channel-first
export const expandChannelFirstShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;      // [seq_len, dim] row-major
@group(0) @binding(1) var<storage, read> cumsum: array<u32>;     // [seq_len] prefix sum of durations
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [dim, total_frames] channel-first

struct Params {
  seq_len: u32,
  dim: u32,
  total_frames: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.total_frames * params.dim;
  if (idx >= total) { return; }

  // Output layout: [dim, total_frames] — idx = channel * total_frames + frame
  let channel = idx / params.total_frames;
  let frame = idx % params.total_frames;

  // Binary search: find token i where cumsum[i-1] <= frame < cumsum[i]
  var lo: u32 = 0u;
  var hi: u32 = params.seq_len;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (cumsum[mid] <= frame) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  let token = lo;

  output[idx] = input[token * params.dim + channel];
}
`;

export const istftShader = /* wgsl */ `
// iSTFT synthesis: conv_post [22, genLength] → waveform [waveformLength]
// Gather-based ConvTranspose: each thread computes one output sample
// Fuses: magnitude/phase split, exp, sin(sin(ph)), cos(sin(ph)), ConvTranspose scatter

@group(0) @binding(0) var<storage, read> conv_post: array<f32>;    // [22, gen_length]
@group(0) @binding(1) var<storage, read> weight_real: array<f32>;  // [11, 20]
@group(0) @binding(2) var<storage, read> weight_imag: array<f32>;  // [11, 20]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [waveform_length]

struct Params {
  gen_length: u32,
  waveform_length: u32,
  bins: u32,       // 11
  kernel_size: u32, // 20
  stride: u32,     // 5
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_pos = gid.x;
  if (out_pos >= params.waveform_length) { return; }

  var sum: f32 = 0.0;

  // For each kernel tap, check if this output position has a contribution
  for (var k: u32 = 0u; k < params.kernel_size; k = k + 1u) {
    if (out_pos < k) { continue; }
    let rem = out_pos - k;
    if (rem % params.stride != 0u) { continue; }
    let t = rem / params.stride;
    if (t >= params.gen_length) { continue; }

    // For each frequency bin, compute magnitude/phase and accumulate
    for (var b: u32 = 0u; b < params.bins; b = b + 1u) {
      let mag_val = conv_post[b * params.gen_length + t];
      let ph_val = conv_post[(b + params.bins) * params.gen_length + t];

      let mag = exp(mag_val);
      let sin_ph = sin(ph_val);
      let real_comp = mag * cos(sin_ph);
      let imag_comp = mag * sin(sin_ph);

      sum += real_comp * weight_real[b * params.kernel_size + k]
           - imag_comp * weight_imag[b * params.kernel_size + k];
    }
  }

  output[out_pos] = sum;
}
`;
