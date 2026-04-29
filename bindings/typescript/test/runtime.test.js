import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { designMatrix, fitModel, loadModelSpec, predict } from "../src/runtime.js";

const fixturesDir = path.resolve("../../tests/fixtures");
const rustRuntimeBin = process.env.MARS_RUNTIME_BIN ?? "";

test("matches checked-in runtime portability fixtures", async () => {
  const files = await readdir(fixturesDir);
  const modelSpecs = files
    .filter((name) => name.startsWith("model_spec_") && name.endsWith(".json"))
    .sort();

  assert.ok(modelSpecs.length > 0);

  for (const file of modelSpecs) {
    const suffix = file.slice("model_spec_".length, -".json".length);
    const spec = loadModelSpec(await readJson(path.join(fixturesDir, file)));
    const fixture = await readJson(
      path.join(fixturesDir, `runtime_portability_fixture_${suffix}.json`),
    );
    const probe = denullMatrix(fixture.probe);
    assertNestedClose(designMatrix(spec, probe), denullMatrix(fixture.design_matrix));
    assertNestedClose(predict(spec, probe), denullVector(fixture.predict));
  }
});

test("declares training as unsupported", () => {
  assert.throws(
    () => fitModel(),
    /training is not supported in @mars-earth\/runtime/i,
  );
});

test(
  "prefers Rust runtime when a binary is available",
  { skip: !rustRuntimeBin },
  () => {
    const spec = loadModelSpec({
      spec_version: "1.0",
      params: {},
      feature_schema: { n_features: 1, feature_names: ["x"] },
      basis_terms: [{ kind: "constant" }],
      coefficients: [2],
    });
    assertNestedClose(designMatrix(spec, [[0], [1]]), [[1], [1]]);
    assertNestedClose(predict(spec, [[0], [1]]), [2, 2]);
  },
);

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

function denullMatrix(rows) {
  return rows.map(denullVector);
}

function denullVector(values) {
  return values.map((value) => (value === null ? Number.NaN : value));
}

function assertNestedClose(actual, expected) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length);
    for (let index = 0; index < expected.length; index += 1) {
      assertNestedClose(actual[index], expected[index]);
    }
    return;
  }
  if (Number.isNaN(expected)) {
    assert.ok(Number.isNaN(actual));
    return;
  }
  assert.ok(Math.abs(actual - expected) <= 1e-12, `${actual} != ${expected}`);
}
