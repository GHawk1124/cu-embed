{
  description = "Rust + CUDA kernels embedded with rust-embed";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;
        craneLib = crane.mkLib pkgs;

        cudaToolkit = pkgs.buildEnv {
          name = "cuda-toolkit-with-nvrtc-static";
          paths = [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cuda_nvrtc.static
          ];
          pathsToLink = [
            "/bin"
            "/include"
            "/lib"
            "/nix-support"
          ];
          ignoreCollisions = true;
        };
        cudaDriverLibPath = "/run/opengl-driver/lib";
        nvccCcbin = lib.getExe' pkgs.cudaPackages.backendStdenv.cc.cc "g++";
        gccStaticLibDir = "${pkgs.cudaPackages.backendStdenv.cc.cc}/lib";

        cuFilter = path: type:
          (builtins.baseNameOf path == "README.md") ||
          (builtins.baseNameOf path == "LICENSE-MIT") ||
          (builtins.baseNameOf path == "LICENSE-APACHE") ||
          (lib.hasSuffix ".cu" path) ||
          (craneLib.filterCargoSources path type);

        src = lib.cleanSourceWith {
          src = ./.;
          filter = cuFilter;
        };

        cudaDeps = [ cudaToolkit ];

        commonArgs = {
          inherit src;
          strictDeps = true;
          nativeBuildInputs = cudaDeps;
          NVCC = "${cudaToolkit}/bin/nvcc";
          NVCC_CCBIN = nvccCcbin;
          CUDA_PATH = cudaToolkit;
          RUSTFLAGS = "-L native=${gccStaticLibDir}";
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
      in
      {
        packages.default = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        packages.example = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          cargoExtraArgs = "--example add_scalar";
          nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.makeWrapper ];
          postFixup = ''
            wrapProgram $out/bin/add_scalar \
              --prefix LD_LIBRARY_PATH : ${cudaDriverLibPath}
          '';
        });

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.rustfmt
          ] ++ cudaDeps;

          NVCC = "${cudaToolkit}/bin/nvcc";
          NVCC_CCBIN = nvccCcbin;
          CUDA_PATH = cudaToolkit;
          RUSTFLAGS = "-L native=${gccStaticLibDir}";

          shellHook = ''
            export LD_LIBRARY_PATH=${cudaDriverLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '';
        };
      });
}
