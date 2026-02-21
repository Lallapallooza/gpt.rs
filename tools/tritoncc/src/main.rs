use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelKind {
    ElementwiseBinaryF32,
    ElementwiseUnaryF32,
    Placeholder,
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

    let kernel_kind = detect_kernel_kind(&source);
    let source_preview = source
        .lines()
        .take(12)
        .map(|line| format!("// {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let ptx = match kernel_kind {
        KernelKind::ElementwiseBinaryF32 => emit_elementwise_binary_ptx(&arch),
        KernelKind::ElementwiseUnaryF32 => emit_elementwise_unary_ptx(&arch),
        KernelKind::Placeholder => emit_placeholder_ptx(&arch, input.display().to_string(), source_preview),
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create output dir {}: {e}", parent.display()))?;
    }
    fs::write(&output, ptx)
        .map_err(|e| format!("failed to write output {}: {e}", output.display()))?;

    let kernel_symbol = match kernel_kind {
        KernelKind::ElementwiseBinaryF32 => "gpt_rs_triton_ewise_binary_f32".to_string(),
        KernelKind::ElementwiseUnaryF32 => "gpt_rs_triton_ewise_unary_f32".to_string(),
        KernelKind::Placeholder => derive_symbol_name(&input),
    };
    let param_abi = match kernel_kind {
        KernelKind::ElementwiseBinaryF32 => {
            vec![
                "*fp32".to_string(),
                "*fp32".to_string(),
                "*fp32".to_string(),
                "u32".to_string(),
                "u32".to_string(),
            ]
        }
        KernelKind::ElementwiseUnaryF32 => {
            vec![
                "*fp32".to_string(),
                "*fp32".to_string(),
                "u32".to_string(),
                "u32".to_string(),
            ]
        }
        KernelKind::Placeholder => vec!["*fp32".to_string(), "*fp32".to_string(), "i32".to_string()],
    };
    let note = match kernel_kind {
        KernelKind::ElementwiseBinaryF32 => "elementwise binary PTX emitted by tritoncc",
        KernelKind::ElementwiseUnaryF32 => "elementwise unary PTX emitted by tritoncc",
        KernelKind::Placeholder => "placeholder metadata emitted by bootstrap tritoncc",
    };
    let meta_doc = KernelMeta {
        schema_version: 1,
        tool: "tritoncc",
        tool_version: "0.1.0",
        arch,
        input: input.display().to_string(),
        output: output.display().to_string(),
        kernel_symbol,
        param_abi,
        shared_mem_bytes: 0,
        num_warps: 4,
        note,
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

fn detect_kernel_kind(source: &str) -> KernelKind {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed == "// gpt_rs.kernel: elementwise_binary_f32" {
            return KernelKind::ElementwiseBinaryF32;
        }
        if trimmed == "// gpt_rs.kernel: elementwise_unary_f32" {
            return KernelKind::ElementwiseUnaryF32;
        }
    }
    KernelKind::Placeholder
}

fn emit_placeholder_ptx(arch: &str, input: String, source_preview: String) -> String {
    format!(
        r#"// tritoncc placeholder PTX artifact
// This artifact is non-executable and exists to lock CLI/metadata contracts.
// arch: {arch}
// input: {input}
.version 8.0
.target {arch}
.address_size 64
// source preview:
{source_preview}
"#
    )
}

fn emit_elementwise_binary_ptx(arch: &str) -> String {
    format!(
        r#".version 7.0
.target {arch}
.address_size 64

.visible .entry gpt_rs_triton_ewise_binary_f32(
    .param .u64 lhs_ptr,
    .param .u64 rhs_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 op
)
{{
    .reg .pred %p<7>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd1, [lhs_ptr];
    ld.param.u64 %rd2, [rhs_ptr];
    ld.param.u64 %rd3, [out_ptr];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [op];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra L_DONE;

    mul.wide.u32 %rd4, %r6, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra L_ADD;
    setp.eq.u32 %p3, %r2, 1;
    @%p3 bra L_SUB;
    setp.eq.u32 %p4, %r2, 2;
    @%p4 bra L_MUL;
    setp.eq.u32 %p5, %r2, 3;
    @%p5 bra L_DIV;
    setp.eq.u32 %p6, %r2, 4;
    @%p6 bra L_MAX;
    bra L_MIN;

L_ADD:
    add.f32 %f3, %f1, %f2;
    bra L_STORE;

L_SUB:
    sub.f32 %f3, %f1, %f2;
    bra L_STORE;

L_MUL:
    mul.f32 %f3, %f1, %f2;
    bra L_STORE;

L_DIV:
    div.full.f32 %f3, %f1, %f2;
    bra L_STORE;

L_MAX:
    max.f32 %f3, %f1, %f2;
    bra L_STORE;

L_MIN:
    min.f32 %f3, %f1, %f2;

L_STORE:
    st.global.f32 [%rd7], %f3;

L_DONE:
    ret;
}}
"#
    )
}

fn emit_elementwise_unary_ptx(arch: &str) -> String {
    format!(
        r#".version 7.0
.target {arch}
.address_size 64

.visible .entry gpt_rs_triton_ewise_unary_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 op
)
{{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [op];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra L_DONE;

    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];

    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra L_NEG;
    bra L_ABS;

L_NEG:
    neg.f32 %f2, %f1;
    bra L_STORE;

L_ABS:
    abs.f32 %f2, %f1;

L_STORE:
    st.global.f32 [%rd5], %f2;

L_DONE:
    ret;
}}
"#
    )
}

fn derive_symbol_name(input: &Path) -> String {
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
