// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use criterion::{criterion_group, criterion_main, Criterion};
use givp::{givp, GivpConfig};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

fn bench_sphere_5d(c: &mut Criterion) {
    let bounds = vec![(-5.12, 5.12); 5];
    c.bench_function("sphere_5d", |b| {
        b.iter(|| {
            let cfg = GivpConfig {
                max_iterations: 20,
                seed: Some(42),
                integer_split: Some(5),
                ..Default::default()
            };
            std::hint::black_box(givp(sphere, &bounds, cfg).unwrap())
        })
    });
}

fn bench_rosenbrock_5d(c: &mut Criterion) {
    let bounds = vec![(-5.0, 10.0); 5];
    c.bench_function("rosenbrock_5d", |b| {
        b.iter(|| {
            let cfg = GivpConfig {
                max_iterations: 20,
                seed: Some(42),
                integer_split: Some(5),
                ..Default::default()
            };
            std::hint::black_box(givp(rosenbrock, &bounds, cfg).unwrap())
        })
    });
}

fn bench_rastrigin_10d(c: &mut Criterion) {
    let bounds = vec![(-5.12, 5.12); 10];
    c.bench_function("rastrigin_10d", |b| {
        b.iter(|| {
            let cfg = GivpConfig {
                max_iterations: 20,
                seed: Some(42),
                integer_split: Some(10),
                ..Default::default()
            };
            std::hint::black_box(givp(rastrigin, &bounds, cfg).unwrap())
        })
    });
}

criterion_group!(
    benches,
    bench_sphere_5d,
    bench_rosenbrock_5d,
    bench_rastrigin_10d
);
criterion_main!(benches);
