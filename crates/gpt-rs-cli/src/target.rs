//! `target`: the targets a program compiles for: the presets, the probed host, and target files.

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use gpt_rs_ir::targets::{probe_host, Target, TargetDesc, PRESETS};

#[derive(ClapArgs)]
pub struct TargetArgs {
    #[command(subcommand)]
    command: TargetCommand,
}

#[derive(Subcommand)]
enum TargetCommand {
    /// Print the reference and the facts, as TOML, of a target: a preset, `host`, or a target file.
    Show {
        #[arg(default_value = "host")]
        target: String,
    },
    /// List the presets, each with its reference and description.
    List,
    /// Probe this machine again, and replace the host target in its cache file. The probe knows
    /// only the Zen 5 desktop parts.
    Probe,
}

pub fn run(args: TargetArgs) -> Result<()> {
    match args.command {
        TargetCommand::Show { target: name } => {
            let target = Target::resolve(&name)?;
            println!("# {}", String::from(target.reference()));
            print!("{}", target.to_toml());
        }
        TargetCommand::List => {
            for (name, _) in PRESETS {
                let desc = TargetDesc::resolve(name)?;
                println!("{name}@{}  {}", desc.fingerprint(), desc.description);
            }
        }
        TargetCommand::Probe => {
            let (desc, path) = probe_host()?;
            println!(
                "# host@{}, written to {}",
                desc.fingerprint(),
                path.display()
            );
            print!("{}", desc.to_toml());
        }
    }
    Ok(())
}
