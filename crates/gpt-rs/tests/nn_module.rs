//! `#[nn::module]`: parameter names, generated sublayer calls and forward arguments.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs::module::{Layer, Module, ParamVisitor, ParamVisitorMut, TensorRole};
use gpt_rs::nn;
use gpt_rs::tensor::{DeviceTensor, DeviceTensorOps, Shape, Tensor as HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

type Backend = CpuPortableBackend;

#[nn::module]
struct Scale {
    weight: Tensor,
}

#[nn::module]
impl Scale {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.mul(&self.weight)
    }
}

#[nn::module]
struct ScaleAdd {
    weight: Tensor,
}

#[nn::module]
impl ScaleAdd {
    fn forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.mul(&self.weight)?.add(b)
    }
}

#[nn::module]
enum Branch {
    #[module(rename = "left")]
    Left(Scale),
    Right(Scale),
}

#[nn::module]
impl Branch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Branch::Left(scale) | Branch::Right(scale) => scale.call(x),
        }
    }
}

const A_LOG: &str = "A_log";

#[nn::module]
struct Net {
    #[module(config)]
    width: usize,
    scale: Scale,
    maybe: Option<Scale>,
    missing: Option<Scale>,
    stack: Vec<Scale>,
    #[module(rename = A_LOG)]
    a_log: Tensor,
    bias: Option<Tensor>,
    no_bias: Option<Tensor>,
    #[module(flatten)]
    branch: Branch,
    pair: ScaleAdd,
}

#[nn::module]
impl Net {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.scale(x)?;
        let y = self.maybe(&y)?.unwrap_or(y);
        let mut y = self.missing(&y)?.unwrap_or(y);
        for scale in &self.stack {
            y = scale.call(&y)?;
        }
        let y = self.branch(&y)?;
        let y = self.pair((&y, x))?;
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }
}

fn backend() -> Arc<Backend> {
    Arc::new(CpuPortableBackend::new())
}

fn filled(backend: &Arc<Backend>, value: f32) -> DeviceTensor<Backend> {
    let host = HostTensor::from_vec(Shape::new([2]), vec![value; 2]).unwrap();
    DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
}

fn scale(backend: &Arc<Backend>, value: f32) -> Scale<Backend> {
    Scale {
        weight: filled(backend, value),
    }
}

fn net(backend: &Arc<Backend>) -> Net<Backend> {
    Net {
        width: 2,
        scale: scale(backend, 2.0),
        maybe: Some(scale(backend, 3.0)),
        missing: None,
        stack: vec![scale(backend, 5.0), scale(backend, 7.0)],
        a_log: filled(backend, 0.0),
        bias: Some(filled(backend, 1.0)),
        no_bias: None,
        branch: Branch::Left(scale(backend, 11.0)),
        pair: ScaleAdd {
            weight: filled(backend, 13.0),
        },
    }
}

fn names<M: Module<Backend>>(module: &M) -> Vec<String> {
    let mut names = Vec::new();
    let mut visit = |name: &str, role: TensorRole, tensor: &DeviceTensor<Backend>| {
        assert_eq!(role, TensorRole::Parameter);
        assert_eq!(tensor.shape().dims(), [2]);
        names.push(name.to_string());
        Ok(())
    };
    module
        .visit_params(&mut ParamVisitor::new(&mut visit))
        .unwrap();
    names
}

#[test]
fn parameters_are_named_by_field_path() {
    let backend = backend();
    let net = net(&backend);
    assert_eq!(net.width, 2);
    assert_eq!(
        names(&net),
        [
            "scale.weight",
            "maybe.weight",
            "stack.0.weight",
            "stack.1.weight",
            "A_log",
            "bias",
            "left.weight",
            "pair.weight",
        ]
    );
    assert_eq!(names(&Branch::Right(scale(&backend, 1.0))), ["weight"]);
}

#[test]
fn mutable_visit_matches_and_updates_parameters() {
    let backend = backend();
    let mut net = net(&backend);
    let mut visited = Vec::new();
    let mut visit = |name: &str, _: TensorRole, tensor: &mut DeviceTensor<Backend>| {
        visited.push(name.to_string());
        *tensor = filled(&backend, 1.0);
        Ok(())
    };
    net.visit_params_mut(&mut ParamVisitorMut::new(&mut visit))
        .unwrap();
    assert_eq!(visited, names(&net));

    // Every scale is 1 and the bias 1: `x * 1 + x + 1`.
    let x = filled(&backend, 4.0);
    let y = net.call(&x).unwrap().to_host().unwrap();
    assert_eq!(y.data(), [9.0, 9.0]);
}

#[test]
fn forward_runs_sublayers_through_generated_calls() {
    let backend = backend();
    let net = net(&backend);
    let x = filled(&backend, 0.5);
    // 0.5 * 2 * 3 * 5 * 7 * 11 * 13 + 0.5 + 1; `missing` is skipped.
    let expected = 0.5 * 2.0 * 3.0 * 5.0 * 7.0 * 11.0 * 13.0 + 0.5 + 1.0;
    assert_eq!(
        net.call(&x).unwrap().to_host().unwrap().data(),
        [expected; 2]
    );
    assert_eq!(
        net.forward(&x).unwrap().to_host().unwrap().data(),
        [expected; 2]
    );

    let pair = ScaleAdd {
        weight: filled(&backend, 3.0),
    };
    let y = pair.call((&x, &filled(&backend, 1.0))).unwrap();
    assert_eq!(y.to_host().unwrap().data(), [2.5, 2.5]);
}

#[test]
fn module_names_are_type_names() {
    assert_eq!(<Net<Backend> as Module<Backend>>::NAME, "Net");
    assert_eq!(<Branch<Backend> as Module<Backend>>::NAME, "Branch");
    assert_eq!(<Scale<Backend> as Module<Backend>>::NAME, "Scale");
}
