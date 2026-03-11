use std::collections::BTreeSet;
use std::env;
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::{
    CubinManifest, KernelManifest, Manifest, ARCH_ENV_VAR, ASSET_ENV_VAR, MANIFEST_FILE_NAME,
    PTX_ARCH_ENV_VAR,
};

const DEFAULT_PTX_BASELINE_ARCH: u32 = 80;

/// Summary of the artifacts emitted by [`Builder::build`].
#[derive(Debug, Clone)]
pub struct BuildOutput {
    pub asset_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub arches: Vec<String>,
    pub kernels: Vec<KernelManifest>,
}

/// Build-time configuration for compiling `.cu` sources into embedded CUDA assets.
///
/// The builder is intended to be called from your crate's `build.rs`. It discovers
/// kernel sources, compiles one CUBIN per selected `sm_*` target, emits a
/// conservative PTX fallback, writes a manifest, and exports
/// `CU_EMBED_ASSET_DIR` for `rust-embed`.
#[derive(Debug, Default, Clone)]
pub struct Builder {
    sources: Vec<PathBuf>,
    source_dirs: Vec<PathBuf>,
    arches: Option<Vec<String>>,
    ptx_arch: Option<String>,
    out_dir: Option<PathBuf>,
    include_dirs: Vec<PathBuf>,
    nvcc_args: Vec<String>,
}

impl Builder {
    /// Creates a new builder with default architecture discovery and PTX settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a single `.cu` source file.
    pub fn source<P: Into<PathBuf>>(mut self, source: P) -> Self {
        self.sources.push(source.into());
        self
    }

    /// Adds multiple `.cu` source files.
    pub fn sources<I, P>(mut self, sources: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.sources.extend(sources.into_iter().map(Into::into));
        self
    }

    /// Recursively adds all `.cu` files under a directory.
    pub fn source_dir<P: Into<PathBuf>>(mut self, source_dir: P) -> Self {
        self.source_dirs.push(source_dir.into());
        self
    }

    /// Overrides the concrete `sm_*` targets used for CUBIN generation.
    pub fn arches<I, S>(mut self, arches: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.arches = Some(arches.into_iter().map(normalize_arch).collect());
        self
    }

    /// Overrides the PTX target used for the fallback artifact.
    pub fn ptx_arch<S: Into<String>>(mut self, arch: S) -> Self {
        self.ptx_arch = Some(normalize_ptx_arch(arch));
        self
    }

    /// Overrides the output directory used for generated assets.
    pub fn out_dir<P: Into<PathBuf>>(mut self, out_dir: P) -> Self {
        self.out_dir = Some(out_dir.into());
        self
    }

    /// Adds an include directory passed through to `nvcc`.
    pub fn include_dir<P: Into<PathBuf>>(mut self, include_dir: P) -> Self {
        self.include_dirs.push(include_dir.into());
        self
    }

    /// Adds a raw additional `nvcc` argument.
    pub fn nvcc_arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.nvcc_args.push(arg.into());
        self
    }

    /// Compiles the configured CUDA sources and emits the embedded asset manifest.
    ///
    /// This writes generated assets under `OUT_DIR/cu-embed` by default and exports
    /// `CU_EMBED_ASSET_DIR` so a `rust-embed` asset type can point at them using
    /// `#[folder = "$CU_EMBED_ASSET_DIR"]`.
    ///
    /// The builder also asks `nvcc` for non-system header dependencies and emits
    /// `cargo:rerun-if-changed` directives for them, so edits to local `.cuh` and
    /// `.h` files trigger rebuilds automatically.
    pub fn build(self) -> Result<BuildOutput, BuildError> {
        emit_cargo_env();

        let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
        let sources = self.resolve_sources()?;
        if sources.is_empty() {
            return Err(BuildError::NoSources);
        }

        let arches = self.resolve_arches(&nvcc)?;
        if arches.is_empty() {
            return Err(BuildError::NoArchitectures);
        }
        let ptx_arch = self.resolve_ptx_arch(&arches)?;

        let mut rerun_paths = BTreeSet::new();
        for source in &sources {
            rerun_paths.insert(source.clone());
            rerun_paths.extend(self.discover_dependencies(&nvcc, source)?);
        }
        for path in &rerun_paths {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        let asset_dir = self.default_output_dir()?;
        fs::create_dir_all(asset_dir.join("cubins"))?;
        fs::create_dir_all(asset_dir.join("ptx"))?;

        let mut names = BTreeSet::new();
        let mut kernels = Vec::new();

        for source in sources {
            let name = source
                .file_stem()
                .and_then(OsStr::to_str)
                .ok_or_else(|| BuildError::InvalidKernelName(source.clone()))?
                .to_owned();

            if !names.insert(name.clone()) {
                return Err(BuildError::DuplicateKernelName(name));
            }

            let mut cubins = Vec::new();
            for arch in &arches {
                let relative = PathBuf::from("cubins")
                    .join(&name)
                    .join(format!("{arch}.cubin"));
                let output_path = asset_dir.join(&relative);
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }

                self.compile_source(&nvcc, &source, arch, &output_path)?;
                cubins.push(CubinManifest {
                    arch: arch.clone(),
                    file: to_unix_path(&relative),
                });
            }

            let ptx_relative = PathBuf::from("ptx").join(format!("{name}.ptx"));
            let ptx_output = asset_dir.join(&ptx_relative);
            self.compile_ptx(&nvcc, &source, &ptx_arch, &ptx_output)?;

            kernels.push(KernelManifest {
                name,
                source: source.display().to_string(),
                ptx: to_unix_path(&ptx_relative),
                cubins,
            });
        }

        let manifest = Manifest {
            version: 1,
            kernels: kernels.clone(),
        };
        let manifest_path = asset_dir.join(MANIFEST_FILE_NAME);
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

        println!("cargo:rustc-env={}={}", ASSET_ENV_VAR, asset_dir.display());

        Ok(BuildOutput {
            asset_dir,
            manifest_path,
            arches,
            kernels,
        })
    }

    fn default_output_dir(&self) -> Result<PathBuf, BuildError> {
        Ok(self.out_dir.clone().unwrap_or_else(|| {
            PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set by Cargo"))
                .join("cu-embed")
        }))
    }

    fn resolve_sources(&self) -> Result<Vec<PathBuf>, BuildError> {
        let mut seen = BTreeSet::new();

        for source in &self.sources {
            seen.insert(source.clone());
        }

        for source_dir in &self.source_dirs {
            collect_cu_files(source_dir, &mut seen)?;
        }

        Ok(seen.into_iter().collect())
    }

    fn resolve_arches(&self, nvcc: &str) -> Result<Vec<String>, BuildError> {
        if let Some(arches) = &self.arches {
            let arches = arches
                .iter()
                .cloned()
                .filter(|arch| !arch.is_empty())
                .collect::<Vec<_>>();
            validate_real_arches(&arches)?;
            return Ok(arches);
        }

        if let Ok(raw) = env::var(ARCH_ENV_VAR) {
            let arches = split_arch_list(&raw);
            if !arches.is_empty() {
                validate_real_arches(&arches)?;
                return Ok(arches);
            }
        }

        discover_arches(nvcc)
    }

    fn resolve_ptx_arch(&self, arches: &[String]) -> Result<String, BuildError> {
        if let Some(ptx_arch) = &self.ptx_arch {
            validate_virtual_arch(ptx_arch)?;
            return Ok(ptx_arch.clone());
        }

        if let Ok(raw) = env::var(PTX_ARCH_ENV_VAR) {
            let raw = raw.trim();
            if !raw.is_empty() {
                let ptx_arch = normalize_ptx_arch(raw);
                validate_virtual_arch(&ptx_arch)?;
                return Ok(ptx_arch);
            }
        }

        let ptx_arch = default_ptx_arch(arches);
        validate_virtual_arch(&ptx_arch)?;
        Ok(ptx_arch)
    }

    fn compile_source(
        &self,
        nvcc: &str,
        source: &Path,
        arch: &str,
        output_path: &Path,
    ) -> Result<(), BuildError> {
        self.invoke_nvcc(nvcc, "-cubin", source, arch, output_path)
    }

    fn compile_ptx(
        &self,
        nvcc: &str,
        source: &Path,
        arch: &str,
        output_path: &Path,
    ) -> Result<(), BuildError> {
        self.invoke_nvcc(nvcc, "-ptx", source, arch, output_path)
    }

    fn discover_dependencies(
        &self,
        nvcc: &str,
        source: &Path,
    ) -> Result<BTreeSet<PathBuf>, BuildError> {
        let depfile = self.default_output_dir()?.join("deps").join(
            source
                .file_name()
                .ok_or_else(|| BuildError::InvalidKernelName(source.to_path_buf()))?,
        );
        let depfile = depfile.with_extension("d");
        if let Some(parent) = depfile.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut command = Command::new(nvcc);
        command
            .arg("-MM")
            .arg("-MF")
            .arg(&depfile)
            .arg("--std=c++14");

        for include_dir in cuda_include_dirs()
            .into_iter()
            .chain(self.include_dirs.iter().cloned())
        {
            command.arg("-I").arg(include_dir);
        }

        if let Ok(ccbin) = env::var("NVCC_CCBIN") {
            command.arg("-ccbin").arg(ccbin);
        }

        for arg in &self.nvcc_args {
            command.arg(arg);
        }

        command.arg(source);
        let output = command.output().map_err(|source_err| BuildError::Command {
            command: format!("{nvcc} -MM"),
            source: source_err,
        })?;
        if !output.status.success() {
            return Err(BuildError::DependencyScanFailed {
                source: source.to_path_buf(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let depfile_contents =
            fs::read_to_string(&depfile).map_err(|error| BuildError::MissingDependencyFile {
                source: source.to_path_buf(),
                depfile: depfile.clone(),
                error,
            })?;
        parse_depfile_paths(&depfile_contents).map_err(|message| {
            BuildError::InvalidDependencyFile {
                source: source.to_path_buf(),
                depfile,
                message,
            }
        })
    }

    fn invoke_nvcc(
        &self,
        nvcc: &str,
        mode: &str,
        source: &Path,
        arch: &str,
        output_path: &Path,
    ) -> Result<(), BuildError> {
        let mut command = Command::new(nvcc);
        command
            .arg(mode)
            .arg(format!("-arch={arch}"))
            .arg("--std=c++14");

        for include_dir in cuda_include_dirs()
            .into_iter()
            .chain(self.include_dirs.iter().cloned())
        {
            command.arg("-I").arg(include_dir);
        }

        if let Ok(ccbin) = env::var("NVCC_CCBIN") {
            command.arg("-ccbin").arg(ccbin);
        }

        for arg in &self.nvcc_args {
            command.arg(arg);
        }

        command.arg(source).arg("-o").arg(output_path);
        let output = command.output().map_err(|source_err| BuildError::Command {
            command: nvcc.to_owned(),
            source: source_err,
        })?;

        if !output.status.success() {
            return Err(BuildError::NvccFailed {
                source: source.to_path_buf(),
                arch: arch.to_owned(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let metadata = fs::metadata(output_path).map_err(|error| BuildError::MissingOutput {
            source: source.to_path_buf(),
            arch: arch.to_owned(),
            output: output_path.to_path_buf(),
            error,
        })?;
        if metadata.len() == 0 {
            return Err(BuildError::EmptyOutput {
                source: source.to_path_buf(),
                arch: arch.to_owned(),
                output: output_path.to_path_buf(),
            });
        }

        Ok(())
    }
}

/// Errors produced while compiling or describing embedded CUDA assets.
#[derive(Debug)]
pub enum BuildError {
    Command {
        command: String,
        source: std::io::Error,
    },
    DuplicateKernelName(String),
    InvalidKernelName(PathBuf),
    Io(std::io::Error),
    Json(serde_json::Error),
    DependencyScanFailed {
        source: PathBuf,
        stdout: String,
        stderr: String,
    },
    InvalidDependencyFile {
        source: PathBuf,
        depfile: PathBuf,
        message: String,
    },
    InvalidArchitecture(String),
    InvalidPtxArchitecture(String),
    MissingDependencyFile {
        source: PathBuf,
        depfile: PathBuf,
        error: std::io::Error,
    },
    MissingOutput {
        source: PathBuf,
        arch: String,
        output: PathBuf,
        error: std::io::Error,
    },
    EmptyOutput {
        source: PathBuf,
        arch: String,
        output: PathBuf,
    },
    NoArchitectures,
    NoSources,
    NvccFailed {
        source: PathBuf,
        arch: String,
        stdout: String,
        stderr: String,
    },
    NvccListFailed(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command { command, source } => {
                write!(f, "failed to execute {command}: {source}")
            }
            Self::DuplicateKernelName(name) => {
                write!(
                    f,
                    "duplicate CUDA kernel name `{name}`; file stems must be unique"
                )
            }
            Self::InvalidKernelName(path) => {
                write!(
                    f,
                    "unable to determine a kernel name from {}",
                    path.display()
                )
            }
            Self::Io(source) => source.fmt(f),
            Self::Json(source) => source.fmt(f),
            Self::DependencyScanFailed {
                source,
                stdout,
                stderr,
            } => write!(
                f,
                "nvcc dependency scan failed for {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                source.display()
            ),
            Self::InvalidDependencyFile {
                source,
                depfile,
                message,
            } => write!(
                f,
                "invalid nvcc dependency file {} for {}: {message}",
                depfile.display(),
                source.display()
            ),
            Self::InvalidArchitecture(arch) => {
                write!(
                    f,
                    "invalid CUDA real architecture `{arch}`; expected `sm_XX` or `sm_XXa`"
                )
            }
            Self::InvalidPtxArchitecture(arch) => {
                write!(
                    f,
                    "invalid CUDA PTX architecture `{arch}`; expected `compute_XX`, `compute_XXa`, or `sm_XX`"
                )
            }
            Self::MissingDependencyFile {
                source,
                depfile,
                error,
            } => write!(
                f,
                "nvcc reported dependencies for {}, but {} was not produced: {error}",
                source.display(),
                depfile.display()
            ),
            Self::MissingOutput {
                source,
                arch,
                output,
                error,
            } => write!(
                f,
                "nvcc reported success for {} on {arch}, but {} was not produced: {error}",
                source.display(),
                output.display()
            ),
            Self::EmptyOutput {
                source,
                arch,
                output,
            } => write!(
                f,
                "nvcc reported success for {} on {arch}, but {} is empty",
                source.display(),
                output.display()
            ),
            Self::NoArchitectures => write!(f, "no CUDA architectures were selected"),
            Self::NoSources => write!(f, "no CUDA source files were provided"),
            Self::NvccFailed {
                source,
                arch,
                stdout,
                stderr,
            } => write!(
                f,
                "nvcc failed for {} on {arch}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                source.display()
            ),
            Self::NvccListFailed(stderr) => {
                write!(f, "nvcc --list-gpu-code failed:\n{stderr}")
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl From<std::io::Error> for BuildError {
    fn from(source: std::io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for BuildError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

fn emit_cargo_env() {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
    println!("cargo:rerun-if-env-changed={ARCH_ENV_VAR}");
    println!("cargo:rerun-if-env-changed={PTX_ARCH_ENV_VAR}");
}

fn discover_arches(nvcc: &str) -> Result<Vec<String>, BuildError> {
    let output = Command::new(nvcc)
        .arg("--list-gpu-code")
        .output()
        .map_err(|source| BuildError::Command {
            command: format!("{nvcc} --list-gpu-code"),
            source,
        })?;

    if !output.status.success() {
        return Err(BuildError::NvccListFailed(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let mut arches = Vec::new();
    let mut seen = BTreeSet::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let arch = line.trim();
        // `nvcc --list-gpu-code` may also report `compute_*` virtual targets.
        // Those are for PTX generation, not the concrete cubin set.
        if arch.starts_with("sm_") && seen.insert(arch.to_owned()) {
            arches.push(arch.to_owned());
        }
    }

    Ok(arches)
}

fn collect_cu_files(dir: &Path, files: &mut BTreeSet<PathBuf>) -> Result<(), BuildError> {
    let mut entries = fs::read_dir(dir)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();

    for path in entries {
        if path.is_dir() {
            collect_cu_files(&path, files)?;
        } else if path.extension() == Some(OsStr::new("cu")) {
            files.insert(path);
        }
    }

    Ok(())
}

fn cuda_include_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(include_dir) = env::var("CUDA_INCLUDE_DIR") {
        dirs.push(PathBuf::from(include_dir));
    }

    for root_var in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Ok(root) = env::var(root_var) {
            dirs.push(PathBuf::from(root).join("include"));
        }
    }

    dirs
}

fn split_arch_list(raw: &str) -> Vec<String> {
    raw.split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|segment| !segment.is_empty())
        .map(normalize_arch)
        .collect()
}

fn normalize_arch<S: Into<String>>(arch: S) -> String {
    arch.into().trim().to_owned()
}

fn normalize_ptx_arch<S: Into<String>>(arch: S) -> String {
    let arch = arch.into().trim().to_owned();
    match arch.strip_prefix("sm_") {
        Some(rest) => format!("compute_{rest}"),
        None => arch,
    }
}

fn default_ptx_arch(arches: &[String]) -> String {
    let mut supported = arches
        .iter()
        .filter_map(|arch| parse_arch_number(arch, "sm_"))
        .collect::<Vec<_>>();

    supported.sort_unstable();
    supported.dedup();

    let selected = supported
        .iter()
        .copied()
        .find(|arch| *arch == DEFAULT_PTX_BASELINE_ARCH)
        .or_else(|| {
            supported
                .iter()
                .copied()
                .filter(|arch| *arch > DEFAULT_PTX_BASELINE_ARCH)
                .min()
        })
        .or_else(|| supported.iter().copied().max())
        .unwrap_or(DEFAULT_PTX_BASELINE_ARCH);

    format!("compute_{selected}")
}

fn to_unix_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn parse_depfile_paths(raw: &str) -> Result<BTreeSet<PathBuf>, String> {
    let normalized = raw.replace("\\\r\n", "").replace("\\\n", "");
    let (_, deps) = normalized
        .split_once(':')
        .ok_or_else(|| "missing dependency rule separator".to_owned())?;

    let mut paths = BTreeSet::new();
    let mut current = String::new();
    let mut escaping = false;

    for ch in deps.chars() {
        if escaping {
            current.push(ch);
            escaping = false;
            continue;
        }

        match ch {
            '\\' => escaping = true,
            ch if ch.is_whitespace() => {
                if !current.is_empty() {
                    paths.insert(PathBuf::from(std::mem::take(&mut current)));
                }
            }
            _ => current.push(ch),
        }
    }

    if escaping {
        return Err("dangling escape in dependency file".to_owned());
    }

    if !current.is_empty() {
        paths.insert(PathBuf::from(current));
    }

    Ok(paths)
}

fn parse_arch_number(arch: &str, prefix: &str) -> Option<u32> {
    let rest = arch.strip_prefix(prefix)?;
    let digit_count = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count < 2 {
        return None;
    }
    rest[..digit_count].parse::<u32>().ok()
}

fn validate_real_arches(arches: &[String]) -> Result<(), BuildError> {
    for arch in arches {
        if !is_valid_arch(arch, "sm_") {
            return Err(BuildError::InvalidArchitecture(arch.clone()));
        }
    }
    Ok(())
}

fn validate_virtual_arch(arch: &str) -> Result<(), BuildError> {
    if is_valid_arch(arch, "compute_") {
        return Ok(());
    }
    Err(BuildError::InvalidPtxArchitecture(arch.to_owned()))
}

fn is_valid_arch(arch: &str, prefix: &str) -> bool {
    let Some(rest) = arch.strip_prefix(prefix) else {
        return false;
    };
    let digit_count = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count < 2 {
        return false;
    }

    let suffix = &rest[digit_count..];
    suffix.is_empty() || suffix == "a"
}
