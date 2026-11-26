// ai: build/format.mjs: the format's arithmetic as the C gives it (src/wasm.c lizard_format_json: the constants, each
// ai: ring's span and modules, each version's picture, tiers, blocks and rooms, each picture's pixels a module), for
// ai: sim/ob.mjs's face, so a page, a worker or a script reads it with no codec instance; and build/blocks.js, the
// ai: blocks a frame by version and a block's payload from the same table as a classic script, which a page loads
// ai: before its first paint (lizard-web/send.html's inline script: the Density label before the modules run).
// ai: Written by build.sh after the module (compile targets of the C, never edited), each whole beside its place and
// ai: renamed over it.
import { renameSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import create from "../bindings/wasm/dist/lizard.mjs";

const at = (name) => fileURLToPath(new URL(`../build/${name}`, import.meta.url));
const put = (out, text) => { writeFileSync(`${out}.part`, text); renameSync(`${out}.part`, out); console.log(`${out}: ${text.length} bytes`); };
const M = await create();
const n = M._lizard_format_json(0, 0), p = M._malloc(n + 1);
if (M._lizard_format_json(p, n + 1) !== n) throw new Error("lizard_format_json: its length moved");
const table = JSON.parse(M.UTF8ToString(p));
put(at("format.mjs"), `// The format's arithmetic as liblizard's C gives it (src/wasm.c lizard_format_json), written by liblizard/build.sh\n// (gen/format.mjs): never edited.\nexport default ${JSON.stringify(table)};\n`);
const blocks = { min: table.VERSION_MIN, payload: table.PAYLOAD, blocks: table.versions.map((v) => v.blocks) };
put(at("blocks.js"), `// The blocks a frame by version (the default senders' split) and a block's payload, as liblizard's C gives them\n// (src/wasm.c lizard_format_json), for a page's script before its first paint; written by liblizard/build.sh\n// (gen/format.mjs): never edited.\nglobalThis.LIZARD_BLOCKS = ${JSON.stringify(blocks)};\n`);
