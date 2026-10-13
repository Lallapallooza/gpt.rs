use anyhow::Result;
use gpt_rs::profiling::WorkStats;
use gpt_rs_ir::ir::TensorType;

use super::utils::{escape_c_string, push_block, sizing};

#[derive(Default)]
pub(super) struct OpProfile {
    ops: Vec<OpProfileEntry>,
}
struct OpProfileEntry {
    label: String,
    signature: String,
    work: WorkStats,
}
impl OpProfile {
    pub(super) fn register(&mut self, label: &str, signature: String, work: WorkStats) -> usize {
        let id = self.ops.len();
        self.ops.push(OpProfileEntry {
            label: label.to_string(),
            signature,
            work,
        });
        id
    }
}
pub(super) fn saturating_u64_from_u128(value: u128) -> u64 {
    if value > u64::MAX as u128 {
        u64::MAX
    } else {
        value as u64
    }
}
pub(super) fn matmul_work_stats(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    lhs_elem_bytes: usize,
    rhs_elem_bytes: usize,
) -> WorkStats {
    let batch = batch as u128;
    let m = m as u128;
    let n = n as u128;
    let k = k as u128;
    let elements = saturating_u64_from_u128(batch.saturating_mul(m).saturating_mul(n));
    let lhs_bytes = batch
        .saturating_mul(m)
        .saturating_mul(k)
        .saturating_mul(lhs_elem_bytes as u128);
    let rhs_bytes = batch
        .saturating_mul(k)
        .saturating_mul(n)
        .saturating_mul(rhs_elem_bytes as u128);
    let out_bytes = batch.saturating_mul(m).saturating_mul(n).saturating_mul(4);
    let flops = batch
        .saturating_mul(m)
        .saturating_mul(n)
        .saturating_mul(k)
        .saturating_mul(2);
    WorkStats {
        elements,
        bytes_read: saturating_u64_from_u128(lhs_bytes.saturating_add(rhs_bytes)),
        bytes_written: saturating_u64_from_u128(out_bytes),
        flops: saturating_u64_from_u128(flops),
        alloc_bytes: 0,
        alloc_count: 0,
    }
}
pub(super) fn elementwise_work_stats(
    out_spec: &TensorType,
    input_specs: &[TensorType],
) -> Result<WorkStats> {
    let (out_elems, out_bytes) = sizing(out_spec)?;
    let mut bytes_read: u64 = 0;
    for spec in input_specs {
        let (_, bytes) = sizing(spec)?;
        bytes_read = bytes_read.saturating_add(bytes as u64);
    }
    Ok(WorkStats {
        elements: out_elems as u64,
        bytes_read,
        bytes_written: out_bytes as u64,
        flops: out_elems as u64,
        alloc_bytes: 0,
        alloc_count: 0,
    })
}
fn emit_u64_array(out: &mut String, name: &str, values: &[u64]) {
    let values_str = if values.is_empty() {
        "0".to_string()
    } else {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let block = format!(
        r#"
            static const uint64_t {name}[GPTRS_C_OP_ARRAY_LEN] = {{{values_str}}};
        "#
    );
    push_block(out, 0, &block);
}

fn emit_str_array(out: &mut String, name: &str, values: &[String]) {
    let values_str = if values.is_empty() {
        "\"\"".to_string()
    } else {
        values
            .iter()
            .map(|value| {
                let escaped = escape_c_string(value);
                format!("\"{escaped}\"")
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let block = format!(
        r#"
            static const char* {name}[GPTRS_C_OP_ARRAY_LEN] = {{{values_str}}};
        "#
    );
    push_block(out, 0, &block);
}

pub(super) fn emit_c_profile_metadata(profile: &OpProfile) -> String {
    let mut out = String::new();
    let op_count = profile.ops.len();
    let header = format!(
        r#"
            #if defined(GPTRS_C_PROFILE)
            #define GPTRS_C_OP_COUNT {op_count}u
            #define GPTRS_C_OP_ARRAY_LEN (GPTRS_C_OP_COUNT ? GPTRS_C_OP_COUNT : 1)
            static uint64_t gpt_rs_c_prof_op_ns[GPTRS_C_OP_ARRAY_LEN];
            static uint64_t gpt_rs_c_prof_op_calls[GPTRS_C_OP_ARRAY_LEN];
        "#
    );
    push_block(&mut out, 0, &header);
    let labels: Vec<String> = profile
        .ops
        .iter()
        .map(|entry| entry.label.clone())
        .collect();
    emit_str_array(&mut out, "gpt_rs_c_prof_op_labels", &labels);
    let elements: Vec<u64> = profile
        .ops
        .iter()
        .map(|entry| entry.work.elements)
        .collect();
    emit_u64_array(&mut out, "gpt_rs_c_prof_op_elements_data", &elements);
    let bytes_read: Vec<u64> = profile
        .ops
        .iter()
        .map(|entry| entry.work.bytes_read)
        .collect();
    emit_u64_array(&mut out, "gpt_rs_c_prof_op_bytes_read_data", &bytes_read);
    let bytes_written: Vec<u64> = profile
        .ops
        .iter()
        .map(|entry| entry.work.bytes_written)
        .collect();
    emit_u64_array(
        &mut out,
        "gpt_rs_c_prof_op_bytes_written_data",
        &bytes_written,
    );
    let flops: Vec<u64> = profile.ops.iter().map(|entry| entry.work.flops).collect();
    emit_u64_array(&mut out, "gpt_rs_c_prof_op_flops_data", &flops);
    let signatures: Vec<String> = profile
        .ops
        .iter()
        .map(|entry| entry.signature.clone())
        .collect();
    emit_str_array(&mut out, "gpt_rs_c_prof_op_signatures", &signatures);
    let helpers = r#"
            void gpt_rs_c_prof_op_reset(void) {
              memset(gpt_rs_c_prof_op_ns, 0, sizeof(gpt_rs_c_prof_op_ns));
              memset(gpt_rs_c_prof_op_calls, 0, sizeof(gpt_rs_c_prof_op_calls));
            }
            size_t gpt_rs_c_prof_op_count(void) { return GPTRS_C_OP_COUNT; }
            const char* gpt_rs_c_prof_op_label(size_t idx) {
              return idx < GPTRS_C_OP_COUNT ? gpt_rs_c_prof_op_labels[idx] : "";
            }
            size_t gpt_rs_c_prof_op_snapshot(uint64_t* ns, uint64_t* calls, size_t len) {
              const size_t count = GPTRS_C_OP_COUNT;
              const size_t copy_len = len < count ? len : count;
              if (ns) { memcpy(ns, gpt_rs_c_prof_op_ns, copy_len * sizeof(uint64_t)); }
              if (calls) { memcpy(calls, gpt_rs_c_prof_op_calls, copy_len * sizeof(uint64_t)); }
              return count;
            }
            const char* gpt_rs_c_prof_op_signature(size_t idx) {
              return idx < GPTRS_C_OP_COUNT ? gpt_rs_c_prof_op_signatures[idx] : "";
            }
            uint64_t gpt_rs_c_prof_op_elements(size_t idx) {
              return idx < GPTRS_C_OP_COUNT ? gpt_rs_c_prof_op_elements_data[idx] : 0;
            }
            uint64_t gpt_rs_c_prof_op_bytes_read(size_t idx) {
              return idx < GPTRS_C_OP_COUNT ? gpt_rs_c_prof_op_bytes_read_data[idx] : 0;
            }
            uint64_t gpt_rs_c_prof_op_bytes_written(size_t idx) {
              return idx < GPTRS_C_OP_COUNT ? gpt_rs_c_prof_op_bytes_written_data[idx] : 0;
            }
            uint64_t gpt_rs_c_prof_op_flops(size_t idx) {
              return idx < GPTRS_C_OP_COUNT ? gpt_rs_c_prof_op_flops_data[idx] : 0;
            }
            #endif
        "#
    .to_string();
    push_block(&mut out, 0, &helpers);
    out
}
pub(super) fn signature_unary(out_spec: &TensorType, input_spec: &TensorType) -> String {
    let input_sig = input_spec.to_string();
    let out_sig = out_spec.to_string();
    format!("x={input_sig} out={out_sig}")
}
pub(super) fn signature_binary(
    out_spec: &TensorType,
    lhs_spec: &TensorType,
    rhs_spec: &TensorType,
) -> String {
    let lhs_sig = lhs_spec.to_string();
    let rhs_sig = rhs_spec.to_string();
    let out_sig = out_spec.to_string();
    format!("lhs={lhs_sig} rhs={rhs_sig} out={out_sig}")
}
pub(super) fn signature_generic(out_spec: &TensorType, input_specs: &[TensorType]) -> String {
    let inputs = input_specs
        .iter()
        .enumerate()
        .map(|(idx, spec)| {
            let sig = spec.to_string();
            format!("in{idx}={sig}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let out_sig = out_spec.to_string();
    let out = format!("out={out_sig}");
    if inputs.is_empty() {
        out
    } else {
        format!("{inputs} {out}")
    }
}
pub(super) fn register_op_profile_unary(
    profile: &mut OpProfile,
    label: &str,
    out_spec: &TensorType,
    input_spec: &TensorType,
) -> Result<usize> {
    let signature = signature_unary(out_spec, input_spec);
    let work = elementwise_work_stats(out_spec, std::slice::from_ref(input_spec))?;
    Ok(profile.register(label, signature, work))
}
pub(super) fn register_op_profile_binary(
    profile: &mut OpProfile,
    label: &str,
    out_spec: &TensorType,
    lhs_spec: &TensorType,
    rhs_spec: &TensorType,
) -> Result<usize> {
    let signature = signature_binary(out_spec, lhs_spec, rhs_spec);
    let inputs = [lhs_spec.clone(), rhs_spec.clone()];
    let work = elementwise_work_stats(out_spec, &inputs)?;
    Ok(profile.register(label, signature, work))
}
pub(super) fn register_op_profile_generic(
    profile: &mut OpProfile,
    label: &str,
    out_spec: &TensorType,
    input_specs: &[TensorType],
) -> Result<usize> {
    let signature = signature_generic(out_spec, input_specs);
    let work = elementwise_work_stats(out_spec, input_specs)?;
    Ok(profile.register(label, signature, work))
}
