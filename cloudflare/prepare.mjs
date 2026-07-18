import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const deploymentDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(deploymentDirectory, "..");
const sourceHtml = resolve(projectDirectory, "zhong2025_reference.html");
const outputDirectory = resolve(deploymentDirectory, "dist");
const outputHtml = resolve(outputDirectory, "neuromatch", "index.html");

await stat(sourceHtml);
await rm(outputDirectory, { recursive: true, force: true });
await mkdir(dirname(outputHtml), { recursive: true });
await copyFile(sourceHtml, outputHtml);

const source = await readFile(sourceHtml, "utf8");
const inlineScripts = [...source.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if (inlineScripts.length !== 1) {
  throw new Error(`Expected exactly one inline script, found ${inlineScripts.length}`);
}
const scriptHash = createHash("sha256")
  .update(inlineScripts[0][1], "utf8")
  .digest("base64");

await writeFile(
  resolve(outputDirectory, "_redirects"),
  "/ /neuromatch/ 302\n",
  "utf8",
);

await writeFile(
  resolve(outputDirectory, "_headers"),
  [
    "/neuromatch/*",
    `  Content-Security-Policy: default-src 'self'; img-src 'self' data:; font-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'sha256-${scriptHash}'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'none'; upgrade-insecure-requests`,
    "  Permissions-Policy: camera=(), microphone=(), geolocation=()",
    "  Referrer-Policy: strict-origin-when-cross-origin",
    "  X-Content-Type-Options: nosniff",
    "  X-Frame-Options: DENY",
    "",
  ].join("\n"),
  "utf8",
);

const sourceSize = (await stat(sourceHtml)).size;
const outputSize = (await stat(outputHtml)).size;

if (sourceSize !== outputSize) {
  throw new Error(`Prepared HTML size mismatch: ${sourceSize} !== ${outputSize}`);
}

console.log(
  JSON.stringify({
    source: sourceHtml,
    output: outputHtml,
    bytes: outputSize,
    route: "/neuromatch/",
    cspScriptHash: `sha256-${scriptHash}`,
  }),
);
