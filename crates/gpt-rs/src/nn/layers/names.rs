//! The module names of a decoder family, which differ between Hugging Face model types.

/// The module names of a decoder family in its Hugging Face checkpoints, for the modules whose
/// names vary between families. The defaults are the names of the Llama family
/// (`LlamaForCausalLM`), which most decoders share.
pub trait DecoderNames: Send + Sync + 'static {
    /// The token embedding.
    const EMBED_TOKENS: &'static str = "embed_tokens";
    /// The learned position embedding of a model with absolute positions.
    const EMBED_POSITIONS: &'static str = "embed_positions";
    /// The list of layers.
    const LAYERS: &'static str = "layers";
    /// The final norm.
    const NORM: &'static str = "norm";
    /// The norm before the token mixer of a layer.
    const INPUT_LAYERNORM: &'static str = "input_layernorm";
    /// The norm before the MLP of a layer.
    const POST_ATTENTION_LAYERNORM: &'static str = "post_attention_layernorm";
    /// The softmax attention of a layer.
    const SELF_ATTN: &'static str = "self_attn";
    /// The projection of an attention that computes the queries, keys and values together.
    const QKV_PROJ: &'static str = "qkv_proj";
    /// The output projection of an attention.
    const O_PROJ: &'static str = "o_proj";
    /// The projection of an MLP to its hidden width.
    const UP_PROJ: &'static str = "up_proj";
    /// The projection of an MLP back from its hidden width.
    const DOWN_PROJ: &'static str = "down_proj";
}

/// The names of the Llama family.
pub struct LlamaNames;

impl DecoderNames for LlamaNames {}
