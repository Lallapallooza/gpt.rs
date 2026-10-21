//! AVX-512 C for the `isa` ops, in the forms of the x86 rules. A group of registers is a C array,
//! written one statement per register; a `reg` buffer that only whole registers reach is an array
//! of vector variables.

use anyhow::{anyhow, bail, ensure, Result};
use gpt_rs_ir::dialects::isa::Access;
use gpt_rs_ir::dialects::x86::Place;
use gpt_rs_ir::dialects::{isa, x86};
use gpt_rs_ir::ir::{Dim, Inst, Level, Type, Value, VectorType};

use super::scalar::{fill, pad_literal};
use super::{strides, Printer};

fn lanes(ty: &VectorType) -> usize {
    *ty.dims.last().expect("a vector has lanes")
}

impl Printer<'_> {
    /// The C of an `isa` op: its group of registers declared, and one statement per register, in
    /// row-major order of the group, which the caller writes.
    pub(super) fn vector(&mut self, inst: Inst) -> Result<Vec<String>> {
        if let Some(access) = Access::of(self.module, inst) {
            return match self.module.op(inst).get::<isa::Prefetch>() {
                Some(prefetch) => self.prefetch(&access, &prefetch.level),
                None => self.memory(inst, &access),
            };
        }
        let op = self.module.op(inst);
        let operands = self.module.operands(inst).to_vec();
        let result = self.module.results(inst)[0];
        let ty = self
            .module
            .ty(result)
            .as_vector()
            .cloned()
            .ok_or_else(|| anyhow!("{} gives no registers", op.name()))?;
        let lanes = lanes(&ty);
        let group = isa::row_major(ty.group());
        let mut statements = Vec::with_capacity(group.len());
        if op.get::<isa::Splat>().is_some() {
            let scalar = self.name(operands[0])?.to_string();
            let form = fill(&x86::splat(ty.dtype, lanes)?.text, &[scalar]);
            statements = vec![form; group.len()];
        } else if op.get::<isa::Expand>().is_some() {
            let input = self.name(operands[0])?.to_string();
            let source = self.vector_type(operands[0])?;
            let source_dims = source.group().to_vec();
            for coords in &group {
                let position = isa::repeated_position(coords, &source_dims);
                statements.push(format!("{input}[{position}]"));
            }
        } else if op.get::<isa::TileMask>().is_some() {
            let extents = self.operand_names(inst)?;
            let mask = x86::mask_type(lanes)?;
            for coords in &group {
                statements.push(tile_mask(coords, &extents, lanes, mask));
            }
        } else if isa::is_lanewise(op) {
            let form = x86::lanewise(self.module, inst)?.text;
            let names = self.operand_names(inst)?;
            for position in 0..group.len() {
                let args: Vec<String> = names
                    .iter()
                    .map(|name| format!("{name}[{position}]"))
                    .collect();
                statements.push(fill(&form, &args));
            }
        } else {
            bail!("{} has no AVX-512 form", op.name());
        }
        self.assign_registers(result, &ty, group.len(), statements)
    }

    /// `isa.prefetch`, `access`: one hint for the cache `level` at the first lane of each register
    /// that a load of its extents reads.
    fn prefetch(&mut self, access: &Access, level: &Level) -> Result<Vec<String>> {
        let name = self.name(access.buffer)?.to_string();
        let hint = self.prefetch_hint(level)?;
        let mut statements = Vec::new();
        for address in self.addresses(access, &name)? {
            statements.push(format!("_mm_prefetch((const char *){address}, {hint});"));
        }
        Ok(statements)
    }

    /// A load, broadcast or store.
    fn memory(&mut self, inst: Inst, access: &Access) -> Result<Vec<String>> {
        let ty = &access.vector;
        let lanes = lanes(ty);
        let group = isa::row_major(ty.group());
        let mask = match access.mask {
            Some(mask) => Some(self.name(mask)?.to_string()),
            None => None,
        };
        let stored = match access.value {
            Some(value) => Some(self.name(value)?.to_string()),
            None => None,
        };
        let buffer = self.name(access.buffer)?.to_string();
        let held = self.registers.contains_key(&access.buffer);
        let (unpadded, place) = match (held, access.pad) {
            (true, _) => (
                self.register_variables(inst, access, &buffer)?,
                Place::Registers,
            ),
            (false, None) => (self.addresses(access, &buffer)?, Place::Memory),
            (false, Some(_)) => (self.addresses(access, &buffer)?, Place::Padded),
        };
        let op = self.module.op(inst);
        let form = x86::memory(op, place, ty.dtype, lanes)?.text;
        let plain_form = match place {
            Place::Padded => x86::memory(op, Place::Memory, ty.dtype, lanes)?.text,
            _ => form.clone(),
        };
        let inside = match access.pad {
            Some(_) => self.inside_lanes(access)?,
            None => Vec::new(),
        };
        let places = match access.pad {
            Some(_) => self.padded_addresses(access, &buffer, &unpadded, &inside)?,
            None => unpadded,
        };
        let pad = match access.pad {
            Some(pad) => pad_literal(ty.dtype, pad.0)?,
            None => String::new(),
        };
        let fills = MemoryFills {
            access,
            form,
            plain_form,
            mask,
            stored,
            pad,
        };
        let mut statements = Vec::with_capacity(group.len());
        for (position, register_place) in places.iter().enumerate() {
            statements.push(fills.register(position, register_place, inside.get(position))?);
        }
        if access.value.is_some() {
            let mut stores = Vec::with_capacity(statements.len());
            for statement in statements {
                stores.push(format!("{statement};"));
            }
            return Ok(stores);
        }
        self.assign_registers(self.module.results(inst)[0], ty, group.len(), statements)
    }

    /// The group of `count` registers of type `ty` that `result` is, declared, and the assignment
    /// of each of `statements` to its register, in order.
    fn assign_registers(
        &mut self,
        result: Value,
        ty: &VectorType,
        count: usize,
        statements: Vec<String>,
    ) -> Result<Vec<String>> {
        let ctype = x86::register_type(ty.dtype, lanes(ty))?;
        let name = self.define(result);
        self.line(format!("{ctype} {name}[{count}];"));
        let mut assignments = Vec::with_capacity(statements.len());
        for (position, statement) in statements.into_iter().enumerate() {
            assignments.push(format!("{name}[{position}] = {statement};"));
        }
        Ok(assignments)
    }

    /// The address of the first lane of each register an access reaches in memory. The offset
    /// holds what moves with the indices' variables, and each address adds the constant part as a
    /// displacement, so accesses at fixed distances from one another share one offset, which the
    /// C compiler holds in one register. A buffer of a 4-bit dtype is addressed in bytes, at the
    /// element offset halved, so each register must start at an even element: every multiple in
    /// the offset and every displacement even.
    fn addresses(&mut self, access: &Access, buffer: &str) -> Result<Vec<String>> {
        let ty = self.buffer_type(access.buffer)?.clone();
        let halved = ty.dtype.bits() == 4;
        let strides = strides(&ty)?;
        let mut constant = 0;
        let mut odd_multiple = false;
        let mut terms = Vec::with_capacity(access.base.len());
        for (&index, &stride) in access.base.iter().zip(&strides) {
            let affine = self.affine_of(index);
            constant += affine.constant * stride as i64;
            for (value, coeff) in affine.terms {
                let multiple = coeff * stride as i64;
                odd_multiple |= multiple % 2 != 0;
                terms.push(format!("{} * {multiple}", self.name(value)?));
            }
        }
        ensure!(
            !(halved && odd_multiple),
            "a register of {} elements may start at an odd element, which the x86 backend \
             cannot address in bytes",
            ty.dtype
        );
        let offset = self.define_offset(&terms);
        let zero = vec![0; access.base.len()];
        let mut addresses = Vec::new();
        for position in access.positions(&zero) {
            let step: i64 = position
                .iter()
                .zip(&strides)
                .map(|(&coord, &stride)| coord * stride as i64)
                .sum();
            let displacement = constant + step;
            ensure!(
                !(halved && displacement % 2 != 0),
                "a register of {} elements starts at an odd element, which the x86 backend \
                 cannot address in bytes",
                ty.dtype
            );
            addresses.push(match halved {
                true => format!("({buffer} + (({offset} + {displacement}) >> 1))"),
                false => format!("({buffer} + {offset} + {displacement})"),
            });
        }
        Ok(addresses)
    }

    /// The address of the first lane of each register of a padded read, of which `places` are
    /// the plain addresses: on each axis that the register's condition checks, a position
    /// outside the buffer counts as 0. So the load reaches no memory outside the buffer, whose
    /// pages may not be mapped and would cost a page walk at each read, while the condition makes
    /// its lanes the pad. The positions it checks do not move inside a register tile's loop, so
    /// the C compiler computes them once outside it.
    fn padded_addresses(
        &self,
        access: &Access,
        buffer: &str,
        places: &[String],
        inside: &[Inside],
    ) -> Result<Vec<String>> {
        let ty = self.buffer_type(access.buffer)?;
        let strides = strides(ty)?;
        let mut names = Vec::with_capacity(access.base.len());
        for &index in &access.base {
            names.push(self.name(index)?.to_string());
        }
        let zero = vec![0; names.len()];
        let mut addresses = Vec::with_capacity(places.len());
        for ((position, place), register_inside) in
            access.positions(&zero).into_iter().zip(places).zip(inside)
        {
            if register_inside.checked.is_empty() {
                addresses.push(place.clone());
                continue;
            }
            let mut terms = Vec::with_capacity(names.len());
            for (axis, name) in names.iter().enumerate() {
                let index = format!("({name} + {})", position[axis]);
                let kept = match (register_inside.checked.contains(&axis), &ty.dims[axis]) {
                    (true, Dim::Static(extent)) => {
                        format!("((uint64_t){index} < {extent} ? {index} : 0)")
                    }
                    _ => index,
                };
                terms.push(format!("{kept} * {}", strides[axis]));
            }
            addresses.push(format!("({buffer} + {})", terms.join(" + ")));
        }
        Ok(addresses)
    }

    /// Where the lanes of each register of a padded read fall inside its buffer. It is a condition
    /// on the register's position on each axis that the read checks for it. For a load, which runs
    /// along the last axis, it is also the run of lanes inside the buffer there, when the read
    /// checks that axis.
    fn inside_lanes(&self, access: &Access) -> Result<Vec<Inside>> {
        let ty = self.buffer_type(access.buffer)?;
        let lanes = lanes(&access.vector);
        let mask = x86::mask_type(lanes)?;
        let mut extents = Vec::with_capacity(ty.dims.len());
        for dim in &ty.dims {
            let Dim::Static(extent) = dim else {
                bail!("the buffer {ty} has a dynamic extent");
            };
            extents.push(*extent);
        }
        let mut names = Vec::with_capacity(access.base.len());
        for &index in &access.base {
            names.push(self.name(index)?.to_string());
        }
        let along = access.along;
        let last = names.len() - 1;
        let zero = vec![0; names.len()];
        let mut inside = Vec::new();
        for (register, position) in access.positions(&zero).into_iter().enumerate() {
            let checks =
                |axis: usize| access.checked.is_empty() || access.checked[register].contains(&axis);
            let mut conditions = Vec::with_capacity(names.len());
            let mut checked = Vec::with_capacity(names.len());
            for axis in 0..names.len() {
                if (along && axis == last) || !checks(axis) {
                    continue;
                }
                let (name, offset, extent) = (&names[axis], position[axis], extents[axis]);
                conditions.push(format!("((uint64_t)({name} + {offset}) < {extent})"));
                checked.push(axis);
            }
            let run = match along && checks(last) {
                true => {
                    let start = format!("{} + {}", names[last], position[last]);
                    let extent = extents[last];
                    Some(format!(
                        "(({mask})({} & ~{}))",
                        prefix(&format!("{extent} - ({start})"), lanes),
                        prefix(&format!("-({start})"), lanes),
                    ))
                }
                false => None,
            };
            inside.push(Inside {
                condition: conditions.join(" & "),
                checked,
                run,
                mask: mask.to_string(),
            });
        }
        Ok(inside)
    }

    /// An offset named once for the registers of one access.
    fn define_offset(&mut self, terms: &[String]) -> String {
        let name = format!("o{}", self.offsets);
        self.offsets += 1;
        let sum = match terms.is_empty() {
            true => "0".to_string(),
            false => terms.join(" + "),
        };
        self.line(format!("int64_t {name} = {sum};"));
        name
    }

    /// The register variable of a `reg` buffer that each register of `access`, the access of the
    /// op of `inst`, is ([`isa::reg_registers`]).
    fn register_variables(
        &mut self,
        inst: Inst,
        access: &Access,
        buffer: &str,
    ) -> Result<Vec<String>> {
        let registers =
            isa::reg_registers(self.module, access, &mut self.forms).map_err(|why| {
                anyhow!(
                    "{} reaches a reg buffer of {}, {why}",
                    self.module.op(inst).name(),
                    self.module.ty(access.buffer)
                )
            })?;
        let mut variables = Vec::with_capacity(registers.len());
        for register in registers {
            variables.push(format!("{buffer}[{register}]"));
        }
        Ok(variables)
    }

    fn vector_type(&self, value: Value) -> Result<VectorType> {
        match self.module.ty(value) {
            Type::Vector(vector) => Ok(vector.clone()),
            other => bail!("{other} is not a group of registers"),
        }
    }
}

/// What the statement of each register of a load, broadcast or store fills the form of its rule
/// with: the names of the access's mask and of the registers it stores when it has them, and the C
/// of its pad.
struct MemoryFills<'a> {
    access: &'a Access,
    form: String,
    /// The form of a register of a padded read whose lanes all lie inside the buffer.
    plain_form: String,
    mask: Option<String>,
    stored: Option<String>,
    pad: String,
}

impl MemoryFills<'_> {
    /// The statement of the register at `position` of the group, at `place`, whose lanes fall
    /// inside the buffer as `inside` says for a padded read. A padded read masks each register by
    /// the flags of its lanes inside, with no branch: registers whose conditions agree would
    /// otherwise branch together, and the C compiler would hold all their loads at once. A
    /// register whose lanes all lie inside wherever the read runs reads as an unpadded one.
    fn register(&self, position: usize, place: &str, inside: Option<&Inside>) -> Result<String> {
        let dtype = self.access.vector.dtype;
        let checks = inside.filter(|register_inside| !register_inside.always());
        let form = match (self.access.pad, checks) {
            (Some(_), None) => &self.plain_form,
            _ => &self.form,
        };
        let flags = checks.map(Inside::flags);
        let mut text = form.replace("$A", place).replace("$R", place);
        if let Some(mask) = &self.mask {
            text = text.replace("$Q", &format!("{mask}[{position}]"));
        }
        if let Some(stored) = &self.stored {
            text = text.replace("$V", &format!("{stored}[{position}]"));
        }
        if let Some(lane_flags) = &flags {
            text = text.replace("$F", lane_flags).replace("$D", &self.pad);
        }
        if text.contains("$B") {
            let mut conditions = Vec::new();
            if let Some(mask) = &self.mask {
                conditions.push(format!("({mask}[{position}] != 0)"));
            }
            if let Some(lane_flags) = &flags {
                conditions.push(format!("(({lane_flags}) != 0)"));
            }
            let fill = match self.access.pad {
                Some(_) => self.pad.clone(),
                None => pad_literal(dtype, 0.0)?,
            };
            let lanes = lanes(&self.access.vector);
            let one = x86::broadcast_one(dtype, lanes, place, &conditions.join(" & "), &fill)?;
            text = text.replace("$B", &one);
        }
        Ok(text)
    }
}

/// Where the lanes of one register of a padded read fall inside its buffer. Where `condition` holds
/// on the positions of the axes `checked`, the lanes inside are those of `run`, or all lanes
/// without a run. An empty condition always holds. `mask` is the C type of the lanes' flags.
struct Inside {
    condition: String,
    /// The axes that `condition` checks.
    checked: Vec<usize>,
    run: Option<String>,
    mask: String,
}

impl Inside {
    /// Whether every lane is inside wherever the read runs: the register checks no axis.
    fn always(&self) -> bool {
        self.condition.is_empty() && self.run.is_none()
    }

    /// The flags of the lanes inside, as a C expression.
    fn flags(&self) -> String {
        let mask = &self.mask;
        let run = match &self.run {
            Some(run) => run.clone(),
            None => format!("(({mask})~0ull)"),
        };
        match self.condition.is_empty() {
            true => run,
            false => format!("({}) ? {run} : ({mask})0", self.condition),
        }
    }
}

/// The flags of the first `count` of `lanes` lanes, as a `uint64_t` C expression: none for a
/// count of 0 or less, every one for `lanes` or more.
fn prefix(count: &str, lanes: usize) -> String {
    let every = match lanes {
        64 => "~0ull".to_string(),
        _ => format!("{:#x}ull", (1u64 << lanes) - 1),
    };
    format!("(({count}) <= 0 ? 0ull : ({count}) >= {lanes} ? {every} : (1ull << ({count})) - 1)")
}

/// The mask of the register at `coords` of a tile mask of `lanes` lanes, inside `extents`.
fn tile_mask(coords: &[usize], extents: &[String], lanes: usize, mask: &str) -> String {
    let Some((&last, outer)) = coords.split_last() else {
        return format!("(({mask}){})", prefix(&extents[0], lanes));
    };
    let mut conditions = Vec::with_capacity(outer.len());
    for (&coord, extent) in outer.iter().zip(extents) {
        conditions.push(format!("({coord} < {extent})"));
    }
    let column = &extents[outer.len()];
    let lane_flags = format!(
        "(({mask}){})",
        prefix(&format!("{column} - {}", last * lanes), lanes)
    );
    match conditions.is_empty() {
        true => lane_flags,
        false => format!("({}) ? {lane_flags} : ({mask})0", conditions.join(" & ")),
    }
}
