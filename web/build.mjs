// esbuild rather than a framework toolchain.
//
// The app is one editor view with decorations on it. A bundler that does
// nothing but resolve imports and emit one file is the whole requirement, and
// this file is short enough to read in full, which a generated config would
// not be.
import * as esbuild from "esbuild";
import { cp, mkdir } from "node:fs/promises";

const watch = process.argv.includes("--watch");

await mkdir("dist", { recursive: true });
await cp("index.html", "dist/index.html");
await cp("src/styles.css", "dist/styles.css");

const options = {
  entryPoints: ["src/main.ts"],
  bundle: true,
  format: "esm",
  target: "es2022",
  outfile: "dist/app.js",
  sourcemap: true,
  minify: !watch,
  logLevel: "info",
};

if (watch) {
  const context = await esbuild.context(options);
  await context.watch();
  console.log("watching for changes");
} else {
  await esbuild.build(options);
}
