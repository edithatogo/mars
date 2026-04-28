# Binding Conformance Harness

This directory defines the shared runtime replay contract for language
bindings.

Bindings consume `manifest.json`, load each `model_spec`, evaluate the fixture
`probe`, and emit parity output JSON:

```json
{
  "binding": "example",
  "fixtures": [
    {
      "name": "v1",
      "design_matrix": [[1.0]],
      "predict": [1.0]
    }
  ]
}
```

Validate an output file with:

```bash
python3 bindings/conformance/runner.py --output path/to/output.json
```

Without `--output`, the runner validates the manifest and expected fixture
schema only.
