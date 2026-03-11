#![doc = include_str!("../README.md")]

mod build_support;
pub mod manifest;
pub mod runtime;

/// Build-time helpers for compiling and embedding CUDA kernels from `build.rs`.
pub mod build {
    pub use crate::build_support::{BuildError, BuildOutput, Builder};
}

pub use runtime::{EmbeddedCudaModules, RuntimeError};
