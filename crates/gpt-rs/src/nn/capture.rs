//! Capture of module outputs, for layer-by-layer comparison with a reference implementation.
//!
//! Outputs are named by their Hugging Face module path. Without a running capture, [`record`] and
//! [`scope`] only test a thread-local slot. They build no names and read back no tensors.

use std::cell::RefCell;
use std::fmt::{self, Write as _};

use anyhow::{ensure, Result};

use crate::backend::PortableBackend;
use crate::tensor::{DeviceTensor, HostTensor};

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct Capture {
    /// Module path of the enclosing scopes.
    path: String,
    /// Width and rows of each module's output, in first-recorded order. Every call appends its
    /// rows, one call per prompt chunk.
    outputs: Vec<(String, usize, Vec<f32>)>,
    /// The first output that could not be read back. [`module_outputs`] returns it.
    error: Option<anyhow::Error>,
}

fn capturing() -> bool {
    CAPTURE.with_borrow(Option::is_some)
}

/// Runs `f` with capture enabled and returns its result with the recorded module outputs, each
/// `[rows, width]`, in first-recorded order.
pub fn module_outputs<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Vec<(String, HostTensor)>)> {
    ensure!(!capturing(), "module output capture is already running");
    /// Ends the capture, also when `f` fails or panics.
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            CAPTURE.set(None);
        }
    }
    CAPTURE.set(Some(Capture::default()));
    let _stop = Stop;
    let value = f()?;
    let capture = CAPTURE.take().unwrap_or_default();
    if let Some(error) = capture.error {
        return Err(error.context("reading back a module output"));
    }
    let outputs = capture
        .outputs
        .into_iter()
        .map(|(name, width, rows)| {
            Ok((
                name,
                HostTensor::f32(vec![rows.len() / width, width], rows)?,
            ))
        })
        .collect::<Result<_>>()?;
    Ok((value, outputs))
}

/// The width and the rows of `output` (`[rows, width]`), read back to the host.
fn read<B: PortableBackend + 'static>(output: &DeviceTensor<B>) -> Result<(usize, Vec<f32>)> {
    let host = output.to_host()?;
    let width = host.dims.last().copied().unwrap_or(1).max(1);
    Ok((width, host.as_f32()?.to_vec()))
}

/// Records `output` (`[rows, width]`) as the output of module `name` in the enclosing scope.
pub fn record<B: PortableBackend + 'static>(name: &str, output: &DeviceTensor<B>) {
    if !capturing() {
        return;
    }
    let read = read(output);
    CAPTURE.with_borrow_mut(|capture| {
        let Some(capture) = capture else { return };
        let (width, values) = match read {
            Ok(read) => read,
            Err(error) => {
                capture.error.get_or_insert(error);
                return;
            }
        };
        let mut path = capture.path.clone();
        if !path.is_empty() && !name.is_empty() {
            path.push('.');
        }
        path.push_str(name);
        match capture.outputs.iter_mut().find(|(seen, ..)| *seen == path) {
            Some((_, _, rows)) => rows.extend_from_slice(&values),
            None => capture.outputs.push((path, width, values)),
        }
    });
}

/// Prefixes module `name` to the names recorded until the returned guard drops.
pub fn scope(name: fmt::Arguments<'_>) -> Scope {
    let parent_len = CAPTURE.with_borrow_mut(|capture| {
        let path = &mut capture.as_mut()?.path;
        let len = path.len();
        if len > 0 {
            path.push('.');
        }
        let _ = path.write_fmt(name);
        Some(len)
    });
    Scope { parent_len }
}

/// Guard of a [`scope`].
pub struct Scope {
    /// Length of the enclosing path, while capturing.
    parent_len: Option<usize>,
}

impl Scope {
    /// Records `output` as the output of the scope's own module.
    pub fn record<B: PortableBackend + 'static>(&self, output: &DeviceTensor<B>) {
        record("", output)
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(len) = self.parent_len {
            CAPTURE.with_borrow_mut(|capture| {
                if let Some(capture) = capture {
                    capture.path.truncate(len);
                }
            });
        }
    }
}
