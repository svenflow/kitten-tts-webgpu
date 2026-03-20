/**
 * ONNX weight parser for Kitten TTS.
 * Loads weights from the ONNX protobuf file and dequantizes them for WebGPU.
 */

// ONNX TensorProto data type enum
const ONNX_FLOAT = 1;
const ONNX_UINT8 = 2;
const ONNX_INT8 = 3;
const ONNX_FLOAT16 = 10;
const ONNX_INT64 = 7;

interface OnnxTensor {
  name: string;
  dims: number[];
  dataType: number;
  rawData: Uint8Array;
}

/**
 * Minimal ONNX protobuf parser.
 * Only parses TensorProto initializers from the model graph.
 * Avoids pulling in a full protobuf library.
 */
export class OnnxParser {
  private buffer: Uint8Array;
  private view: DataView;

  constructor(buffer: ArrayBuffer) {
    this.buffer = new Uint8Array(buffer);
    this.view = new DataView(buffer);
  }

  /** Parse all initializer tensors from the ONNX model. */
  parseInitializers(): Map<string, OnnxTensor> {
    const tensors = new Map<string, OnnxTensor>();

    // ONNX ModelProto: field 7 = graph (GraphProto)
    // GraphProto: field 5 = initializer (repeated TensorProto)
    // TensorProto: field 1 = dims, field 2 = data_type, field 4 = raw_data, field 8 = name

    const graphData = this.findField(this.buffer, 0, this.buffer.length, 7);
    if (!graphData) {
      throw new Error('Could not find graph in ONNX model');
    }

    // Find all initializer fields (field 5) in the graph
    let offset = graphData.start;
    while (offset < graphData.end) {
      const field = this.readTag(offset);
      if (!field) break;

      if (field.fieldNumber === 5 && field.wireType === 2) {
        // Length-delimited: TensorProto
        const len = this.readVarint(field.dataStart);
        const tensorStart = len.end;
        const tensorEnd = tensorStart + len.value;

        const tensor = this.parseTensorProto(tensorStart, tensorEnd);
        if (tensor && tensor.name) {
          tensors.set(tensor.name, tensor);
        }

        offset = tensorEnd;
      } else {
        offset = this.skipField(field);
      }
    }

    return tensors;
  }

  private parseTensorProto(start: number, end: number): OnnxTensor | null {
    let name = '';
    const dims: number[] = [];
    let dataType = 0;
    let rawData: Uint8Array | null = null;
    let floatData: Float32Array | null = null;
    let int32Data: Int32Array | null = null;
    let int64Data: BigInt64Array | null = null;

    let offset = start;
    while (offset < end) {
      const field = this.readTag(offset);
      if (!field) break;

      switch (field.fieldNumber) {
        case 1: // dims (repeated int64)
          if (field.wireType === 0) {
            const v = this.readVarint(field.dataStart);
            dims.push(v.value);
            offset = v.end;
          } else if (field.wireType === 2) {
            // Packed repeated
            const len = this.readVarint(field.dataStart);
            let pos = len.end;
            const packEnd = pos + len.value;
            while (pos < packEnd) {
              const v = this.readVarint(pos);
              dims.push(v.value);
              pos = v.end;
            }
            offset = packEnd;
          } else {
            offset = this.skipField(field);
          }
          break;

        case 2: // data_type
          {
            const v = this.readVarint(field.dataStart);
            dataType = v.value;
            offset = v.end;
          }
          break;

        case 4: // float_data (repeated float, packed)
          if (field.wireType === 2) {
            const len = this.readVarint(field.dataStart);
            const dataStart = len.end;
            // Copy to aligned buffer (protobuf data may not be 4-byte aligned)
            const floatBytes = this.buffer.slice(dataStart, dataStart + len.value);
            floatData = new Float32Array(floatBytes.buffer, 0, len.value / 4);
            offset = dataStart + len.value;
          } else {
            offset = this.skipField(field);
          }
          break;

        case 5: // int32_data (repeated int32, packed)
          if (field.wireType === 2) {
            const len = this.readVarint(field.dataStart);
            const dataStart = len.end;
            const packEnd5 = dataStart + len.value;
            // Packed int32 uses varint encoding (NOT raw 4-byte), decode each value
            const int32Values: number[] = [];
            let pos5 = dataStart;
            while (pos5 < packEnd5) {
              const v = this.readVarint(pos5);
              int32Values.push(v.value | 0); // Convert to signed int32
              pos5 = v.end;
            }
            int32Data = new Int32Array(int32Values);
            offset = packEnd5;
          } else if (field.wireType === 0) {
            // Single varint-encoded int32 value (non-packed)
            const v = this.readVarint(field.dataStart);
            int32Data = new Int32Array([v.value]);
            offset = v.end;
          } else {
            offset = this.skipField(field);
          }
          break;

        case 7: // int64_data (packed)
          if (field.wireType === 2) {
            const len = this.readVarint(field.dataStart);
            const dataStart = len.end;
            // Copy to aligned buffer (protobuf data may not be 8-byte aligned)
            const int64Bytes = this.buffer.slice(dataStart, dataStart + len.value);
            int64Data = new BigInt64Array(int64Bytes.buffer, 0, len.value / 8);
            offset = dataStart + len.value;
          } else {
            offset = this.skipField(field);
          }
          break;

        case 8: // name
          {
            const len = this.readVarint(field.dataStart);
            const nameBytes = this.buffer.subarray(len.end, len.end + len.value);
            name = new TextDecoder().decode(nameBytes);
            offset = len.end + len.value;
          }
          break;

        case 9: // raw_data (field 9 in ONNX TensorProto)
          {
            const len = this.readVarint(field.dataStart);
            rawData = this.buffer.subarray(len.end, len.end + len.value);
            offset = len.end + len.value;
          }
          break;

        default:
          offset = this.skipField(field);
      }
    }

    if (!name) return null;

    // Resolve raw data
    let data: Uint8Array;
    if (rawData) {
      data = rawData;
    } else if (floatData) {
      data = new Uint8Array(floatData.buffer, floatData.byteOffset, floatData.byteLength);
    } else if (int32Data) {
      data = new Uint8Array(int32Data.buffer, int32Data.byteOffset, int32Data.byteLength);
    } else if (int64Data) {
      data = new Uint8Array(int64Data.buffer, int64Data.byteOffset, int64Data.byteLength);
    } else {
      data = new Uint8Array(0);
    }

    return { name, dims, dataType, rawData: data };
  }

  private readTag(offset: number): { fieldNumber: number; wireType: number; dataStart: number } | null {
    if (offset >= this.buffer.length) return null;
    const v = this.readVarint(offset);
    const tag = v.value;
    return {
      fieldNumber: tag >>> 3,
      wireType: tag & 0x7,
      dataStart: v.end,
    };
  }

  private readVarint(offset: number): { value: number; end: number } {
    let value = 0;
    let shift = 0;
    let pos = offset;
    while (pos < this.buffer.length) {
      const byte = this.buffer[pos];
      value |= (byte & 0x7f) << shift;
      pos++;
      if ((byte & 0x80) === 0) break;
      shift += 7;
      if (shift > 35) break; // Prevent infinite loop
    }
    return { value, end: pos };
  }

  private skipField(field: { wireType: number; dataStart: number }): number {
    switch (field.wireType) {
      case 0: // Varint
        return this.readVarint(field.dataStart).end;
      case 1: // 64-bit
        return field.dataStart + 8;
      case 2: { // Length-delimited
        const len = this.readVarint(field.dataStart);
        return len.end + len.value;
      }
      case 5: // 32-bit
        return field.dataStart + 4;
      default:
        throw new Error(`Unknown wire type: ${field.wireType}`);
    }
  }

  private findField(buf: Uint8Array, start: number, end: number, targetField: number): { start: number; end: number } | null {
    let offset = start;
    while (offset < end) {
      const field = this.readTag(offset);
      if (!field) break;

      if (field.fieldNumber === targetField && field.wireType === 2) {
        const len = this.readVarint(field.dataStart);
        return { start: len.end, end: len.end + len.value };
      }

      offset = this.skipField(field);
    }
    return null;
  }
}

/**
 * Dequantize INT8 weights to float32.
 * ONNX uses: float_val = (int8_val - zero_point) * scale
 */
export function dequantizeInt8(
  quantized: Int8Array,
  scale: Float32Array,
  zeroPoint: Int8Array | null,
  shape: number[]
): Float32Array {
  const output = new Float32Array(quantized.length);
  const zp = zeroPoint ? zeroPoint[0] : 0;
  const s = scale[0];

  for (let i = 0; i < quantized.length; i++) {
    output[i] = (quantized[i] - zp) * s;
  }
  return output;
}

/**
 * Dequantize UINT8 weights to float32.
 */
export function dequantizeUint8(
  quantized: Uint8Array,
  scale: Float32Array,
  zeroPoint: Uint8Array | null,
  shape: number[]
): Float32Array {
  const output = new Float32Array(quantized.length);
  const zp = zeroPoint ? zeroPoint[0] : 0;
  const s = scale[0];

  for (let i = 0; i < quantized.length; i++) {
    output[i] = (quantized[i] - zp) * s;
  }
  return output;
}

/**
 * Convert float16 to float32.
 */
export function float16ToFloat32(f16: Uint16Array): Float32Array {
  const output = new Float32Array(f16.length);
  for (let i = 0; i < f16.length; i++) {
    const h = f16[i];
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;

    if (exp === 0) {
      // Subnormal or zero
      output[i] = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
    } else if (exp === 31) {
      // Inf or NaN
      output[i] = frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      output[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
  }
  return output;
}

/**
 * Parse NPZ file (NumPy compressed archive) for voice embeddings.
 * NPZ is just a ZIP file containing .npy files.
 */
export async function parseNpz(buffer: ArrayBuffer): Promise<Map<string, { shape: number[]; data: Float32Array }>> {
  const zip = new Uint8Array(buffer);
  const result = new Map<string, { shape: number[]; data: Float32Array }>();

  let offset = 0;
  while (offset < zip.length - 4) {
    // Look for local file header signature: PK\x03\x04
    if (zip[offset] !== 0x50 || zip[offset + 1] !== 0x4b ||
        zip[offset + 2] !== 0x03 || zip[offset + 3] !== 0x04) {
      break;
    }

    const view = new DataView(buffer, offset);
    const compressionMethod = view.getUint16(8, true);
    let compressedSize = view.getUint32(18, true);
    const fileNameLen = view.getUint16(26, true);
    const extraLen = view.getUint16(28, true);

    const fileName = new TextDecoder().decode(zip.subarray(offset + 30, offset + 30 + fileNameLen));

    // Handle ZIP64 extended information in extra field
    if (compressedSize === 0xFFFFFFFF) {
      // Parse extra field for ZIP64 extended information (header ID 0x0001)
      let extraOffset = offset + 30 + fileNameLen;
      const extraEnd = extraOffset + extraLen;
      while (extraOffset + 4 <= extraEnd) {
        const extraView = new DataView(buffer, extraOffset);
        const headerId = extraView.getUint16(0, true);
        const dataSize = extraView.getUint16(2, true);
        if (headerId === 0x0001 && dataSize >= 16) {
          // ZIP64: uncompressed size (8 bytes), compressed size (8 bytes)
          // Read as Number (safe up to 2^53)
          const lo = extraView.getUint32(12, true);
          const hi = extraView.getUint32(16, true);
          compressedSize = lo + hi * 0x100000000;
          break;
        }
        extraOffset += 4 + dataSize;
      }
    }

    const dataStart = offset + 30 + fileNameLen + extraLen;

    if (fileName.endsWith('.npy') && compressionMethod === 0) {
      // Parse .npy format
      const npyData = zip.subarray(dataStart, dataStart + compressedSize);
      const parsed = parseNpy(npyData.buffer, npyData.byteOffset);
      const name = fileName.replace('.npy', '');
      result.set(name, parsed);
    }

    offset = dataStart + compressedSize;
  }

  return result;
}

function parseNpy(buffer: ArrayBuffer, byteOffset: number): { shape: number[]; data: Float32Array } {
  const result = parseNpyGeneric(buffer, byteOffset);
  return { shape: result.shape, data: result.data };
}

/** Parse a .npy file, handling float32, float16, and int64 dtypes. Always returns Float32Array. */
export function parseNpyGeneric(buffer: ArrayBuffer, byteOffset: number = 0): { shape: number[]; data: Float32Array; dtype: string } {
  const bytes = new Uint8Array(buffer, byteOffset);

  // Magic: \x93NUMPY
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4e) {
    throw new Error('Invalid .npy magic number');
  }

  const majorVersion = bytes[6];
  let headerLen: number;
  let headerStart: number;

  if (majorVersion === 1) {
    headerLen = new DataView(buffer, byteOffset + 8).getUint16(0, true);
    headerStart = 10;
  } else {
    headerLen = new DataView(buffer, byteOffset + 8).getUint32(0, true);
    headerStart = 12;
  }

  const headerStr = new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen));
  const dataStart = headerStart + headerLen;

  // Parse header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (400, 256), }
  const shapeMatch = headerStr.match(/shape['"]\s*:\s*\(([^)]*)\)/);
  const shape = shapeMatch
    ? shapeMatch[1].split(',').map(s => s.trim()).filter(s => s).map(Number)
    : [];

  const descrMatch = headerStr.match(/descr['"]\s*:\s*'([^']*)'/);
  const dtype = descrMatch ? descrMatch[1] : '<f4';

  const totalElements = shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1);

  let data: Float32Array;

  if (dtype === '<f4' || dtype === '=f4' || dtype === 'float32') {
    // float32
    const srcBytes = new Uint8Array(buffer, byteOffset + dataStart, totalElements * 4);
    const aligned = new Uint8Array(totalElements * 4);
    aligned.set(srcBytes);
    data = new Float32Array(aligned.buffer);
  } else if (dtype === '<f2' || dtype === '=f2' || dtype === 'float16') {
    // float16 → convert to float32
    const srcBytes = new Uint8Array(buffer, byteOffset + dataStart, totalElements * 2);
    const aligned = new Uint8Array(totalElements * 2);
    aligned.set(srcBytes);
    const f16 = new Uint16Array(aligned.buffer);
    data = float16ToFloat32(f16);
  } else if (dtype === '<i8' || dtype === '=i8' || dtype === 'int64') {
    // int64 → convert to float32 (lossy but fine for comparisons)
    const srcBytes = new Uint8Array(buffer, byteOffset + dataStart, totalElements * 8);
    const aligned = new Uint8Array(totalElements * 8);
    aligned.set(srcBytes);
    const view = new DataView(aligned.buffer);
    data = new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      // Read as int32 (lower 32 bits) — safe for values < 2^31
      data[i] = view.getInt32(i * 8, true);
    }
  } else {
    throw new Error(`Unsupported .npy dtype: ${dtype}`);
  }

  return { shape, data, dtype };
}
