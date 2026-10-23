# Kernel performance work

`gpt-rs-cli perf` times and profiles the ops, layers and blocks of GPT-2 small, ResNet-18 and
MobileNetV2 at the shapes those models run them. It finds where a backend loses time inside a
model. `scripts/eval.py --workload bench` times whole models ([testing.md](testing.md)).

Source: [crates/gpt-rs-cli/src/perf/](../crates/gpt-rs-cli/src/perf/).

## The cases

`gpt-rs-cli perf list` prints them, over a hundred, named `<model>/<what>` and most with a shape:

- `resnet18/...`: the stem, each stage's conv layers (conv, batch norm, ReLU), shortcut, residual
  and basic block, the pool and the classifier, at 224x224;
- `mobilenet_v2/...`: the stem, each stage's 1x1 expansion, depthwise 3x3, projection and inverted
  residual, the head, at 224x224 with TensorFlow `SAME` padding;
- `gpt2/...`: decode after 128 and 512 tokens and prefill of 128 and 512: the attention layer, the
  decoder layer, the projections, the MLP, the norms, the embeddings and the head;
- `kernel/...`: matmuls, a GEMV, softmax, layer norm, transpose and elementwise ops on their own.

A case of a layer or a block builds the `nn` layers that its model uses, with random weights
loaded as parameters. It builds them with the builders that the Torch parity cases share
(`gpt_rs_backend_tests::layers`), so the passes that fold and outline parameters run as they do for
a loaded model. Each run records the case in a fresh arena, as a model's forward does.

## Timing

```bash
cargo build --release -p gpt-rs-cli --features conversion-c
target/release/gpt-rs-cli perf run --filter resnet18/layer4 --backends x86,c --json out.json
```

For each case and backend it prints:

- the median time the backend ran the case's programs, which a trace sink measures around
  `PortableBackend::run`, so it leaves out recording and planning;
- the first backend's time against each other backend's;
- the first backend's GFLOP/s, from the program's cost analysis, and the GB/s of its arguments and
  results.

`--json` keeps every measurement, among them the call time with recording included.
`--case <name>` picks cases by full name, and `--filter` by part of it.

## Timing a lowered program

```bash
target/release/gpt-rs-cli perf ptir --ptir dump/passes/00012_schedule.ptir --after schedule
```

It times a PTIR program that a backend's passes lowered as far as one of them, such as a pass's
dump edited by hand. The passes after that one lower the program. Its load-time functions run on
random parameters, and its run-time function on random arguments.

## Profiling down to the line of Rust

```bash
target/release/gpt-rs-cli perf profile 'resnet18/layer1.conv1 3x3/1 64>64 @56'
```

It runs the case under `perf record`, then attributes the samples. It needs `perf` and
`addr2line`. The x86 backend compiles its C with debug info. Beside each C file of its cache, it
keeps the line map `program_<hash>.map.json`, which gives for each line the op it comes from, its
number, its location and its trail. So each sample in the code that the x86 backend generated
goes to:

- its line of C;
- the op that wrote it, and the op each pass made that op from;
- the line of Rust and the layer that recorded it.

The tables group the samples by function, line of C, op, op at each level and line of Rust. The
rest of the samples go to their symbols, such as the workers waiting for work. Under
`--backend c`, every sample goes to its symbol.

`perf run --dump-dir` also dumps the module after every pass of the backend's lowering, and the C
with its line map. Look up the op numbers of a trail in the dump of each pass
([ops.md](ops.md#tracing-and-dumps)).
