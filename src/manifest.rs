use serde::{Deserialize, Serialize};

pub const ASSET_ENV_VAR: &str = "CU_EMBED_ASSET_DIR";
pub const ARCH_ENV_VAR: &str = "CU_EMBED_CUDA_ARCHES";
pub const PTX_ARCH_ENV_VAR: &str = "CU_EMBED_PTX_ARCH";
pub const MANIFEST_FILE_NAME: &str = "cu-embed-manifest.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub kernels: Vec<KernelManifest>,
}

impl Manifest {
    pub fn kernel(&self, name: &str) -> Option<&KernelManifest> {
        self.kernels.iter().find(|kernel| kernel.name == name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelManifest {
    pub name: String,
    pub source: String,
    pub ptx: String,
    pub cubins: Vec<CubinManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubinManifest {
    pub arch: String,
    pub file: String,
}
