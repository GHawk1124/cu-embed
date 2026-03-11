# cu-embed

`cu-embed` compiles CUDA `.cu` sources with `nvcc` during `build.rs`, embeds the generated artifacts into your Rust binary, and loads the best module for the installed NVIDIA GPU at runtime.

It is designed to work well with `cudarc`, `rust-embed`, and Nix-based builds.

## What it does

- Compiles one or more `.cu` files into per-architecture CUBINs.
- Generates a conservative PTX fallback for forward compatibility.
- Tracks local `.cuh` and `.h` dependencies automatically through `nvcc -MM`.
- Writes a manifest describing the embedded assets.
- Lets your runtime prefer exact-match CUBINs, then compatible CUBINs, then PTX.

The target machine only needs a suitable NVIDIA driver. It does not need a matching CUDA toolkit installed.

## Build-time usage

In your crate's `Cargo.toml`:

```toml
[build-dependencies]
cu-embed = "0.1"
```

Then in `build.rs`:

```ignore
fn main() {
    cu_embed::build::Builder::new()
        .source_dir("src/kernels")
        .build()
        .expect("failed to build embedded CUDA kernels");
}
```

## Runtime usage

In your runtime crate's `Cargo.toml`, `rust-embed` must have the
`interpolate-folder-path` feature enabled:

```toml
[dependencies]
rust-embed = { version = "8", features = ["interpolate-folder-path"] }
```

Your runtime crate needs a `rust-embed` asset type pointed at `$CU_EMBED_ASSET_DIR`:

```ignore
use cu_embed::EmbeddedCudaModules;
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "$CU_EMBED_ASSET_DIR"]
struct EmbeddedCudaAssets;
```

Then load modules through `EmbeddedCudaModules`:

```ignore
let modules = EmbeddedCudaModules::<EmbeddedCudaAssets>::new()?;
let module = modules.load_module(&ctx, "my_kernel")?;
```

`EmbeddedCudaModules` selects artifacts in this order:

1. exact `sm_*` CUBIN
2. same-major lower-minor CUBIN
3. embedded PTX fallback

## Configuration

- `CU_EMBED_CUDA_ARCHES`: override the CUBIN architectures, for example `sm_80,sm_90`.
- `CU_EMBED_PTX_ARCH`: override the PTX target, for example `compute_80`.
- `NVCC`, `NVCC_CCBIN`, `CUDA_PATH`: standard CUDA toolchain overrides used by the builder.

## Example

See [examples/add_scalar.rs](examples/add_scalar.rs) for a complete runtime example.

## License

Licensed under either of:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.
