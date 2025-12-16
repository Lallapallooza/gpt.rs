pub fn emit_c_kernels() -> String {
    concat!(
        include_str!("preamble.inc.c"),
        include_str!("pack.inc.c"),
        include_str!("compute.inc.c"),
        include_str!("ukernel_16.inc.c"),
        include_str!("ukernel_32.inc.c"),
        include_str!("ukernel_48.inc.c"),
        include_str!("ukernel_64.inc.c"),
        include_str!("ukernel_misc.inc.c"),
        include_str!("api.inc.c")
    )
    .to_string()
}
