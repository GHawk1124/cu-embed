use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaModule, DriverError},
    nvrtc::Ptx,
};
use rust_embed::RustEmbed;

use crate::manifest::{CubinManifest, Manifest, MANIFEST_FILE_NAME};

/// Loads embedded CUDA kernels and selects the best artifact for a device.
///
/// `E` must be a `rust-embed` asset type pointed at the generated asset directory:
/// your runtime crate must enable `rust-embed`'s `interpolate-folder-path`
/// feature for `$CU_EMBED_ASSET_DIR` interpolation to work.
///
/// ```ignore
/// use rust_embed::RustEmbed;
///
/// #[derive(RustEmbed)]
/// #[folder = "$CU_EMBED_ASSET_DIR"]
/// struct EmbeddedCudaAssets;
/// ```
pub struct EmbeddedCudaModules<E> {
    manifest: Manifest,
    _marker: PhantomData<E>,
}

impl<E> EmbeddedCudaModules<E>
where
    E: RustEmbed,
{
    /// Reads the embedded manifest from `E`.
    pub fn new() -> Result<Self, RuntimeError> {
        let manifest_file = E::get(MANIFEST_FILE_NAME).ok_or(RuntimeError::MissingManifest)?;
        let manifest = serde_json::from_slice::<Manifest>(manifest_file.data.as_ref())?;
        Ok(Self {
            manifest,
            _marker: PhantomData,
        })
    }

    /// Returns the embedded kernel manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Loads a single kernel module for the given CUDA context.
    ///
    /// The loader prefers an exact-match CUBIN, then a same-major lower-minor CUBIN,
    /// and finally the embedded PTX fallback.
    pub fn load_module(
        &self,
        ctx: &Arc<CudaContext>,
        kernel_name: &str,
    ) -> Result<Arc<CudaModule>, RuntimeError> {
        match self.select_artifact(ctx, kernel_name)? {
            SelectedArtifact::Cubin(cubin) => {
                let embedded = E::get(&cubin.file)
                    .ok_or_else(|| RuntimeError::MissingAsset(cubin.file.clone()))?;
                Ok(ctx.load_module(Ptx::from_binary(embedded.data.into_owned()))?)
            }
            SelectedArtifact::Ptx(path) => {
                let embedded =
                    E::get(&path).ok_or_else(|| RuntimeError::MissingAsset(path.clone()))?;
                let ptx = String::from_utf8(embedded.data.into_owned())
                    .map_err(|source| RuntimeError::InvalidPtx(path.clone(), source))?;
                Ok(ctx.load_module(Ptx::from_src(ptx))?)
            }
        }
    }

    /// Loads every kernel named in the embedded manifest.
    pub fn load_all_modules(
        &self,
        ctx: &Arc<CudaContext>,
    ) -> Result<BTreeMap<String, Arc<CudaModule>>, RuntimeError> {
        let mut modules = BTreeMap::new();
        for kernel in &self.manifest.kernels {
            modules.insert(kernel.name.clone(), self.load_module(ctx, &kernel.name)?);
        }
        Ok(modules)
    }

    fn select_artifact(
        &self,
        ctx: &Arc<CudaContext>,
        kernel_name: &str,
    ) -> Result<SelectedArtifact, RuntimeError> {
        let kernel = self
            .manifest
            .kernel(kernel_name)
            .ok_or_else(|| RuntimeError::KernelNotFound(kernel_name.to_owned()))?;

        let (major, minor) = ctx.compute_capability()?;
        let device_arch = format!("sm_{major}{minor}");

        if let Some(exact) = kernel.cubins.iter().find(|cubin| cubin.arch == device_arch) {
            return Ok(SelectedArtifact::Cubin(exact.clone()));
        }

        let device_numeric = parse_numeric_arch(&device_arch);
        let fallback = device_numeric.and_then(|device| {
            kernel
                .cubins
                .iter()
                .filter_map(|cubin| {
                    parse_numeric_arch(&cubin.arch).and_then(|candidate| {
                        (candidate.major == device.major && candidate.minor <= device.minor)
                            .then_some((candidate.minor, cubin))
                    })
                })
                .max_by_key(|(minor, _)| *minor)
                .map(|(_, cubin)| cubin.clone())
        });

        Ok(fallback
            .map(SelectedArtifact::Cubin)
            .unwrap_or_else(|| SelectedArtifact::Ptx(kernel.ptx.clone())))
    }
}

/// Errors produced while loading embedded CUDA assets at runtime.
#[derive(Debug)]
pub enum RuntimeError {
    Driver(DriverError),
    InvalidManifest(serde_json::Error),
    InvalidPtx(String, std::string::FromUtf8Error),
    KernelNotFound(String),
    MissingAsset(String),
    MissingManifest,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(source) => source.fmt(f),
            Self::InvalidManifest(source) => source.fmt(f),
            Self::InvalidPtx(path, source) => {
                write!(f, "embedded PTX `{path}` is not valid UTF-8: {source}")
            }
            Self::KernelNotFound(kernel) => write!(f, "unknown embedded CUDA kernel `{kernel}`"),
            Self::MissingAsset(path) => write!(f, "embedded CUDA asset `{path}` is missing"),
            Self::MissingManifest => write!(f, "embedded CUDA manifest is missing"),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(source) => Some(source),
            Self::InvalidManifest(source) => Some(source),
            Self::InvalidPtx(_, source) => Some(source),
            Self::KernelNotFound(_) | Self::MissingAsset(_) | Self::MissingManifest => None,
        }
    }
}

impl From<DriverError> for RuntimeError {
    fn from(source: DriverError) -> Self {
        Self::Driver(source)
    }
}

impl From<serde_json::Error> for RuntimeError {
    fn from(source: serde_json::Error) -> Self {
        Self::InvalidManifest(source)
    }
}

#[derive(Clone, Copy)]
struct NumericArch {
    major: u32,
    minor: u32,
}

enum SelectedArtifact {
    Cubin(CubinManifest),
    Ptx(String),
}

fn parse_numeric_arch(arch: &str) -> Option<NumericArch> {
    let rest = arch.strip_prefix("sm_")?;
    let digit_count = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count < 2 {
        return None;
    }

    let digits = &rest[..digit_count];
    let value = digits.parse::<u32>().ok()?;
    Some(NumericArch {
        major: value / 10,
        minor: value % 10,
    })
}
