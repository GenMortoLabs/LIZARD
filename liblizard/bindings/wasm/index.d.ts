// ai: Types for index.mjs, liblizard's WebAssembly binding (the C API in ../../include/lizard.h says what each does).

/** Loads the module once. build: "auto" (the SIMD build where the engine has WebAssembly SIMD and wasm's exception
 *  handling, else the MVP build), "simd" or "mvp". The first call decides; calls made while it loads share it. */
export declare function init(moduleOptions?: object, options?: { build?: "auto" | "simd" | "mvp" }): Promise<unknown>;
/** Whether this engine has WebAssembly SIMD, which the default build needs with wasm's exception handling (else init
 *  loads the MVP build). */
export declare function hasSimd(): boolean;
export declare class LizardError extends Error {
  readonly name: "LizardError";
  /** One of lizard.h's LIZ_E_* codes (negative). */
  readonly code: number;
}

export declare const BLOCK: 472, ID_BYTES: 12, PAYLOAD: 460, MIN_BLOCKS: 1, MAX_BLOCKS: 67, RINGS: 3, RING_DEFAULT: -1, GAP_MODULES: 12;
export type PixelFormat = "grey" | "rgbx" | "rgba" | "bgra";

export declare function version(): string;
/** Whether the codec's vector paths (WebAssembly SIMD) are in the loaded build: false in the MVP build. */
export declare function simd(): boolean;

export interface Format {
  /** The format's size, 1 to 67 (LIZARD-16 to LIZARD-1072, 16 x the size); a symbol carries the rate profile's blocks
   *  (frameBlocks over codes). */
  blocks: number;
  /** 0 to 2 (the 64, 128 or 256 ring), -1 the default (the 128). */
  ring?: number;
  /** The pictures a second the sender paints, 1 to 240 (default 60), which every frame's word states. */
  fps?: number;
  /** 1, or 2 symbols side by side (default 1). */
  codes?: 1 | 2;
}
export interface Geometry {
  n: number; span: number; pxm: number; side: number; width: number; height: number; gap: number; frameBlocks: number;
}
export declare function ringCells(ring?: number): number;
export declare function geometry(format: Format): Geometry;
export declare function roomFor(blocks: number, ring?: number): number;
/** The blocks a symbol of that size carries (the rate profile's), 0 for a size no format has. */
export declare function blocksFor(blocks: number): number;
export declare function pick(w: number, h: number, options?: { codes?: 1 | 2; ring?: number; top?: number }): number;

export declare class Encoder {
  constructor(format: Format);
  readonly format: Required<Format>;
  readonly geometry: Geometry;
  /** blocks: frameBlocks x 472 bytes; picture: the frame's count (its value mod 4 the pilots). */
  paint<T extends Uint8Array | Uint8ClampedArray = Uint8ClampedArray>(blocks: Uint8Array, picture: number, out?: T, fmt?: PixelFormat): T;
  free(): void;
}

export interface Decoded {
  /** Blocks verified, and their bytes (count x 472). */
  count: number;
  blocks: Uint8Array;
  found: boolean;
  ring: number;
  n: number;
  word: boolean;
  blocksPerSymbol: number;
  total: number;
  /** Corners in the image: x y x y ..., top left, top right, bottom right, bottom left. */
  quad: number[];
  pilot: { subch: number; r: number[]; sd: number[] };
}
export interface ImageLike { data: Uint8Array | Uint8ClampedArray; width: number; height: number }
export declare class Decoder {
  constructor(nmax?: number);
  readonly maxBlocks: number;
  decode(px: Uint8Array | Uint8ClampedArray | ImageLike, w?: number, h?: number,
         options?: { fmt?: PixelFormat; stride?: number }): Decoded;
  free(): void;
}
export declare function layoutRects(w: number, h: number, layout?: 1 | 2): { x: number; y: number; w: number; h: number }[];
export declare function streamFill(id: number): Uint8Array;

export interface TxInfo {
  test: boolean; length: number; sourceBlocks: number; cycle: number; root: string; seed: number; sbLog2: number;
}
export declare class Tx {
  static file(bytes: Uint8Array, name?: string, zstd?: boolean): Tx;
  static test(firstId?: number): Tx;
  next(n: number): Uint8Array;
  readonly info: TxInfo;
  free(): void;
}

export interface Verdict { seen: number; bad: number; judged: number; fresh: number; test: boolean }
export interface Progress {
  header: boolean; done: boolean; length: number; bytesIn: number; fraction: number;
  sourceBlocks: number; verified: number; rejected: number; solveMs: number; forwarded: number; doneAt: number; refused: number; over: number;
}
export declare class Rx {
  constructor(options?: { maxBytes?: number });
  frame(blocks: Uint8Array): Verdict;
  readonly progress: Progress;
  /** Each source block: 0 to 99 the share of its source symbols in, 255 verified. */
  readonly sourceBlocks: Uint8Array;
  readonly name: string;
  readonly root: string;
  readonly error: string;
  data(): Uint8Array | null;
  clear(): void;
  free(): void;
}

export declare class Sender {
  constructor(source: { bytes: Uint8Array; name?: string; zstd?: boolean } | { test: true; firstId?: number }, format: Format);
  readonly geometry: Geometry;
  readonly info: TxInfo;
  picture: number;
  frame<T extends Uint8Array | Uint8ClampedArray = Uint8ClampedArray>(fmt?: PixelFormat, out?: T): { data: T; width: number; height: number; picture: number };
  free(): void;
}
export declare class Receiver {
  constructor(options?: { layout?: 1 | 2; nmax?: number; maxBytes?: number });
  readonly decoder: Decoder;
  readonly rx: Rx;
  push(px: Uint8Array | Uint8ClampedArray | ImageLike, w?: number, h?: number, options?: { fmt?: PixelFormat; stride?: number }):
    { decoded: Decoded[]; verdict: Verdict; progress: Progress };
  readonly file: { name: string; root: string; bytes: Uint8Array } | null;
  clear(): void;
  free(): void;
}
