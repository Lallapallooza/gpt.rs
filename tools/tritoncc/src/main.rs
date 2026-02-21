use std::env;
use std::fs;
use std::path::PathBuf;

struct KernelMeta {
    schema_version: u32,
    tool: &'static str,
    tool_version: &'static str,
    arch: String,
    input: String,
    output: String,
    kernel_symbol: String,
    param_abi: Vec<String>,
    shared_mem_bytes: u32,
    num_warps: u32,
    note: &'static str,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        print_help();
        return Ok(());
    };

    match cmd.as_str() {
        "--help" | "-h" | "help" => {
            print_help();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("tritoncc 0.1.0");
            Ok(())
        }
        "compile" => run_compile(args.collect()),
        "inspect" => run_inspect(args.collect()),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn run_compile(raw_args: Vec<String>) -> Result<(), String> {
    let mut arch: Option<String> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut meta: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--arch" => {
                i += 1;
                arch = raw_args.get(i).cloned();
            }
            "--in" => {
                i += 1;
                input = raw_args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                output = raw_args.get(i).map(PathBuf::from);
            }
            "--meta" => {
                i += 1;
                meta = raw_args.get(i).map(PathBuf::from);
            }
            flag => return Err(format!("unknown compile flag '{flag}'")),
        }
        i += 1;
    }

    let arch = arch.ok_or_else(|| "missing required --arch".to_string())?;
    let input = input.ok_or_else(|| "missing required --in".to_string())?;
    let output = output.ok_or_else(|| "missing required --out".to_string())?;
    let meta = meta.ok_or_else(|| "missing required --meta".to_string())?;

    let source = fs::read_to_string(&input)
        .map_err(|e| format!("failed to read input {}: {e}", input.display()))?;

    let source_preview = source
        .lines()
        .take(12)
        .map(|line| format!("// {line}"))
        .collect::<Vec<_>>()
        .join("\n");

    let ptx = format!(
        r#"// tritoncc placeholder PTX artifact
// This artifact is non-executable and exists to lock CLI/metadata contracts.
// arch: {arch}
// input: {input}
.version 8.0
.target sm_80
.address_size 64
// source preview:
{source_preview}
"#,
        arch = arch,
        input = input.display(),
        source_preview = source_preview
    );

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create output dir {}: {e}", parent.display()))?;
    }
    fs::write(&output, ptx)
        .map_err(|e| format!("failed to write output {}: {e}", output.display()))?;

    let kernel_symbol = derive_symbol_name(&input);
    let meta_doc = KernelMeta {
        schema_version: 1,
        tool: "tritoncc",
        tool_version: "0.1.0",
        arch,
        input: input.display().to_string(),
        output: output.display().to_string(),
        kernel_symbol,
        param_abi: vec!["*fp32".to_string(), "*fp32".to_string(), "i32".to_string()],
        shared_mem_bytes: 0,
        num_warps: 4,
        note: "placeholder metadata emitted by bootstrap tritoncc",
    };

    if let Some(parent) = meta.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create metadata dir {}: {e}", parent.display()))?;
    }
    fs::write(&meta, encode_meta_json(&meta_doc))
        .map_err(|e| format!("failed to write metadata {}: {e}", meta.display()))?;

    println!("compiled {} -> {}", input.display(), output.display());
    println!("metadata {}", meta.display());
    Ok(())
}

fn run_inspect(raw_args: Vec<String>) -> Result<(), String> {
    if raw_args.is_empty() {
        return Err("inspect requires input path".to_string());
    }
    let input = PathBuf::from(&raw_args[0]);
    let source = fs::read_to_string(&input)
        .map_err(|e| format!("failed to read input {}: {e}", input.display()))?;
    println!("input={}", input.display());
    println!("bytes={}", source.len());
    println!("lines={}", source.lines().count());
    println!("symbol={}", derive_symbol_name(&input));
    Ok(())
}

fn derive_symbol_name(input: &PathBuf) -> String {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kernel");
    let mut symbol = String::from("triton_kernel_");
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            symbol.push(ch);
        } else {
            symbol.push('_');
        }
    }
    symbol
}

fn encode_meta_json(meta: &KernelMeta) -> String {
    let abi = meta
        .param_abi
        .iter()
        .map(|item| format!("\"{}\"", json_escape(item)))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"{{
  "schema_version": {schema_version},
  "tool": "{tool}",
  "tool_version": "{tool_version}",
  "arch": "{arch}",
  "input": "{input}",
  "output": "{output}",
  "kernel_symbol": "{kernel_symbol}",
  "param_abi": [{abi}],
  "shared_mem_bytes": {shared_mem_bytes},
  "num_warps": {num_warps},
  "note": "{note}"
}}
"#,
        schema_version = meta.schema_version,
        tool = json_escape(meta.tool),
        tool_version = json_escape(meta.tool_version),
        arch = json_escape(&meta.arch),
        input = json_escape(&meta.input),
        output = json_escape(&meta.output),
        kernel_symbol = json_escape(&meta.kernel_symbol),
        abi = abi,
        shared_mem_bytes = meta.shared_mem_bytes,
        num_warps = meta.num_warps,
        note = json_escape(meta.note)
    )
}

fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn print_help() {
    println!("tritoncc 0.1.0");
    println!("Usage:");
    println!("  tritoncc compile --arch <sm_xx> --in <kernel.triton> --out <kernel.ptx> --meta <kernel.meta.json>");
    println!("  tritoncc inspect <kernel.triton>");
    println!("  tritoncc version");
}
