use std::error::Error;
use std::io;

use cu_embed::EmbeddedCudaModules;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "$CU_EMBED_ASSET_DIR"]
struct EmbeddedCudaAssets;

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("CUDA device 0 is unavailable: {err}");
            return Ok(());
        }
    };

    let stream = ctx.default_stream();
    let modules = EmbeddedCudaModules::<EmbeddedCudaAssets>::new()?;
    let module = modules.load_module(&ctx, "add_scalar").map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("failed to load embedded CUDA module: {err}"),
        )
    })?;
    let function = module.load_function("add_scalar")?;

    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let scalar = 2.5_f32;
    let len = input.len() as i32;
    let expected: Vec<f32> = input.iter().map(|value| value + scalar).collect();

    let mut values = stream.clone_htod(&input)?;
    let mut launch_args = stream.launch_builder(&function);
    launch_args.arg(&mut values);
    launch_args.arg(&scalar);
    launch_args.arg(&len);
    unsafe { launch_args.launch(LaunchConfig::for_num_elems(input.len() as u32)) }?;

    let output: Vec<f32> = stream.clone_dtoh(&values)?;
    assert_eq!(output, expected);
    println!("kernel output: {output:?}");

    Ok(())
}
