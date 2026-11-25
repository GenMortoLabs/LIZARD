// The kernels' Slang as slangc reads it. Each kernel is a .slang file in src/ (with the pieces kernels share in
// src/lib/): Slang, its constants named and defined by the kernel's module, its shared pieces by #include. A kernel's
// source is that file with every #include "<path>" line replaced by the file it names (src/ relative), an
// #include "gen/<name>" by the text the module generates under that name (the unrolled transforms and the like, which
// the preprocessor cannot write), and the module's constants as #define lines after the comments that open the file.
// The result is one text: gen names a module by its text, so every piece and constant is in it.
// ai: Each file is included where the kernel's own file says, once each: the order of the pieces is the order of the
// ai: declarations. A kernel's file can be checked alone with slangc given the defines (-D) and -I src.
import { readFileSync } from "node:fs";

const SRC = new URL("./src/", import.meta.url);
const files = new Map();
const read = (path) => {
  if (!files.has(path)) files.set(path, readFileSync(new URL(path, SRC), "utf8"));
  return files.get(path);
};

const INCLUDE = /^[ \t]*#include "([^"]+)"[ \t]*(\/\/[^\n]*)?$/gm;

// ai: defs: { NAME: value } (a value as Slang text: a number's literal with its suffix where the code wants one, an
// ai: identifier or a short expression); gens: { name: text } for #include "gen/<name>"
export function slang(path, defs = {}, gens = {}) {
  const expand = (text, from, depth) => {
    if (depth > 8) throw new Error(`slang ${path}: includes nested past 8 at ${from}`);
    return text.replace(INCLUDE, (_, f) => {
      if (f.startsWith("gen/")) {
        const g = gens[f.slice(4)];
        if (g === undefined) throw new Error(`slang ${path}: ${from} includes ${f}, which its module does not generate`);
        return g;
      }
      return expand(read(f), f, depth + 1).replace(/\n$/, "");
    });
  };
  const body = expand(read(path), path, 0);
  // ai: the defines after the head (the comment and blank lines that open the file)
  const head = body.match(/^(?:[ \t]*\/\/[^\n]*\n|[ \t]*\n)*/)[0];
  return head + defines(defs) + body.slice(head.length);
}

// ai: the #define lines for defs, in their order
export function defines(defs) {
  let s = "";
  for (const [k, v] of Object.entries(defs)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(k)) throw new Error(`slang: ${k} is no macro name`);
    s += `#define ${k} ${String(v)}\n`;
  }
  return s;
}
