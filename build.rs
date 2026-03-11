#[allow(dead_code)]
#[path = "src/build_support.rs"]
mod build_support;
#[allow(dead_code)]
#[path = "src/manifest.rs"]
mod manifest;

fn main() {
    build_support::Builder::new()
        .source_dir("src/kernels")
        .build()
        .expect("failed to build embedded CUDA kernels");
}
