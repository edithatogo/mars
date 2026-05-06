use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pymars_runtime::{design_matrix, load_model_spec_path, predict};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust-runtime should sit under the repository root")
        .to_path_buf()
}

fn runtime_benchmarks(c: &mut Criterion) {
    let spec_path = repo_root().join("tests/fixtures/model_spec_v1.json");
    let spec = load_model_spec_path(&spec_path).unwrap_or_else(|error| {
        panic!(
            "failed to load benchmark fixture {}: {error}",
            spec_path.display()
        )
    });
    let rows = vec![
        vec![0.0, 0.0, 0.1],
        vec![0.5, 0.25, 0.2],
        vec![1.0, 0.75, 0.3],
        vec![1.5, 1.0, 0.4],
    ];

    c.bench_function("design_matrix_v1", |bench| {
        bench.iter(|| {
            design_matrix(black_box(&spec), black_box(&rows))
                .expect("benchmark design_matrix should validate the fixture spec")
        })
    });

    c.bench_function("predict_v1", |bench| {
        bench.iter(|| {
            predict(black_box(&spec), black_box(&rows))
                .expect("benchmark predict should validate the fixture spec")
        })
    });
}

criterion_group!(benches, runtime_benchmarks);
criterion_main!(benches);
