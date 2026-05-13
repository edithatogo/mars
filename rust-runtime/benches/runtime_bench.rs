use std::env;
use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pymars_runtime::{design_matrix, load_model_spec_path, predict};

const RUNTIME_THREAD_ENV_VAR: &str = "MARS_EARTH_RUNTIME_THREADS";
const BENCH_MARKS: [(usize, &str); 3] = [(64, "small"), (1_024, "medium"), (8_192, "large")];

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

    for (rows_count, size_label) in BENCH_MARKS {
        let rows = make_rows(rows_count);
        add_design_matrix_bench(c, &spec, &rows, size_label, 1);
        add_design_matrix_bench(c, &spec, &rows, size_label, 4);
        add_predict_bench(c, &spec, &rows, size_label, 1);
        add_predict_bench(c, &spec, &rows, size_label, 4);
    }
}

criterion_group!(benches, runtime_benchmarks);
criterion_main!(benches);

fn add_design_matrix_bench(
    c: &mut Criterion,
    spec: &pymars_runtime::ModelSpec,
    rows: &[Vec<f64>],
    size_label: &str,
    thread_count: usize,
) {
    let name = format!(
        "design_matrix_v1_{size_label}_threads_{thread_count}",
        size_label = size_label,
        thread_count = thread_count
    );
    let _thread_guard = ThreadOverrideGuard::new(thread_count);
    c.bench_function(&name, |bench| {
        bench.iter(|| {
            design_matrix(black_box(spec), black_box(rows))
                .expect("benchmark design_matrix should validate the fixture spec")
        })
    });
}

fn add_predict_bench(
    c: &mut Criterion,
    spec: &pymars_runtime::ModelSpec,
    rows: &[Vec<f64>],
    size_label: &str,
    thread_count: usize,
) {
    let name = format!(
        "predict_v1_{size_label}_threads_{thread_count}",
        size_label = size_label,
        thread_count = thread_count
    );
    let _thread_guard = ThreadOverrideGuard::new(thread_count);
    c.bench_function(&name, |bench| {
        bench.iter(|| {
            predict(black_box(spec), black_box(rows))
                .expect("benchmark predict should validate the fixture spec")
        })
    });
}

fn make_rows(rows: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|index| {
            let base = index as f64 / rows as f64;
            vec![base, 0.5 + 0.25 * base.sin(), (base * 10.0).fract()]
        })
        .collect()
}

struct ThreadOverrideGuard {
    previous: Option<String>,
}

impl ThreadOverrideGuard {
    fn new(thread_count: usize) -> Self {
        let previous = env::var(RUNTIME_THREAD_ENV_VAR).ok();
        env::set_var(RUNTIME_THREAD_ENV_VAR, thread_count.to_string());
        ThreadOverrideGuard { previous }
    }
}

impl Drop for ThreadOverrideGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            env::set_var(RUNTIME_THREAD_ENV_VAR, previous);
        } else {
            env::remove_var(RUNTIME_THREAD_ENV_VAR);
        }
    }
}
