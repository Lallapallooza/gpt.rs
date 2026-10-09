use proc_macro::TokenStream;

mod functional;
mod nn_module;

/// Declares a neural-network module in the style of a PyTorch `nn.Module`.
///
/// The macro is re-exported as `gpt_rs::nn::module`. Apply it to the struct (or enum) and to its
/// impls:
///
/// ```ignore
/// #[nn::module]
/// pub struct Mlp {
///     up_proj: Linear,
///     act: Activation,
///     down_proj: Linear,
/// }
///
/// #[nn::module]
/// impl Mlp {
///     fn new(p: &mut Params<B>, act: ActivationFunction) -> Result<Self> {
///         Ok(Self {
///             up_proj: Linear::new(p)?,
///             act: act.layer(),
///             down_proj: Linear::new(p)?,
///         })
///     }
///
///     fn forward(&self, x: &Tensor) -> Tensor {
///         self.down_proj(&self.act(&self.up_proj(x)))
///     }
/// }
/// ```
///
/// # Backend
///
/// The macro adds the backend type parameter `B: PortableBackend + 'static` to the type and to each
/// of its impls, before any other type parameter (`Mlp<B>`). Every field is backend-generic
/// unless it has `#[module(config)]`:
/// - Write a sublayer type without its backend argument. The macro adds `B` before its other
///   arguments, also inside `Option<_>` and `Vec<_>`, so `Linear` becomes `Linear<B>` and
///   `DecoderBlock<N>` becomes `DecoderBlock<B, N>`.
/// - A `Tensor` field is a parameter of type `DeviceTensor<B>`.
///
/// `Tensor` also means `DeviceTensor<B>` in `config` fields and in the impl. Other backend-generic
/// types name `B` explicitly (`&DecodeKvCache<B>`). A layer without parameters uses `B` through a
/// `config` field. For example, `Activation` holds a `fn(&Tensor) -> Tensor`.
///
/// # Parameters
///
/// The type implements `Module<B>`, and its `NAME` is the type's name. The visitors walk the
/// fields in declaration order and visit each field under its field name. For names that do not
/// follow Rust's snake_case field naming, use `#[module(rename = "A_log")]`. The visitors handle
/// fields as follows:
/// - A `Tensor` is the parameter `name`. The parameters of a sublayer are under `name.`.
/// - The visitors skip an `Option<_>` that is `None`.
/// - Element `i` of a `Vec<_>` is under `name.i`.
/// - A `#[module(flatten)]` sublayer adds no `name.` segment.
/// - `#[module(config)]` fields (dimensions, epsilons, flags) are not visited.
///
/// An enum holds one sublayer per variant, `Variant(Layer)`. The visitors visit the active
/// variant, under its `#[module(rename = ...)]` if it has one. A `rename` takes any `&str`
/// expression, such as a string literal or a constant.
///
/// # Construction
///
/// In an impl function that takes a `&mut Params<..>` argument (`gpt_rs::nn::Params`, `p` below),
/// each field initializer of a `Self { .. }` literal that uses `p` runs with `p` inside that
/// field's parameter path. So does the payload of a `Self::Variant(..)`. This is the same path
/// that the visitors use. A tensor field reads its tensor with `p.param()`, whose shape comes from
/// the checkpoint. `p.each(items, |p, item| ..)` builds a `Vec<_>`. A sublayer built outside such
/// an initializer reads from the enclosing path, so build each one inside its field, or enter a
/// path explicitly with `p.scoped(..)`.
///
/// # Forward and sublayer calls
///
/// An impl's `fn forward(&self, a: A, ...) -> T` implements `Layer<B>` with
/// `Output = T` and `Args<'a> = A`. With several arguments, `Args` is the tuple `(A, ...)`.
/// Elided lifetimes become `'a`. The other items of the impl stay inherent.
///
/// Each plain or optional sublayer field gets a method with the same name. The method runs the
/// sublayer with `Layer::call`, which opens the sublayer's profiler layer scope `NAME` around its
/// `forward`. Examples are `self.up_proj(x)`, and `self.self_attn((x, cache, positions))` for
/// several arguments. The method of an `Option<_>` field returns `Option<T>`, which is `None`
/// when the sublayer is absent. `Vec<_>` elements and enum variants run with
/// `layer.call(args)`.
///
/// A sublayer field cannot be named `forward`, `call`, `visit_params`, or `visit_params_mut`. A
/// sublayer field with the same name as another method of the module is a duplicate definition.
#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    match nn_module::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Declares a portable functional. Its body validates the inputs, then computes with the ops of
/// `DeviceTensor`, which record into the lazy graph of the inputs.
///
/// The macro is re-exported as `gpt_rs::functional`:
///
/// ```ignore
/// #[functional]
/// pub fn gelu(x: &Tensor) -> Tensor {
///     ensure_rank_at_least!(x, 1);
///     let half = 0.5 * x;
///     half * (1.0 + (x / 2.0f32.sqrt()).erf())
/// }
/// ```
///
/// # Backend
///
/// The macro adds the backend type parameter `B: PortableBackend + 'static` to the function.
/// `Tensor` means `DeviceTensor<B>` in its signature and body. The function takes no backend
/// argument because its tensors carry the backend. Other backend-generic types name `B`
/// explicitly (`&LinearAttentionCache<B>`).
///
/// # Validation
///
/// The body sees the constant `FUNCTIONAL`, which holds the function's name. The validation macros
/// panic with it in front of their messages, for example `gelu: x must have rank >= 1, got []`.
/// The function is `#[track_caller]`, so the panic names the line that called it.
///
/// # Profiling
///
/// The body runs inside the profiler scope `FUNCTIONAL`, which times the validation and the
/// recording of its ops. The ops run later, under their backend scopes.
#[proc_macro_attribute]
pub fn functional(attr: TokenStream, item: TokenStream) -> TokenStream {
    match functional::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
