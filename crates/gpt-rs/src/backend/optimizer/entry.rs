use std::collections::HashMap;
use std::marker::PhantomData;

use anyhow::{anyhow, Result};

use crate::backend::spec::{Function, ValueId, ValueType};
use crate::tensor::InputRole;

#[derive(Clone, Debug)]
pub struct EntryParam {
    pub id: ValueId,
    pub ty: ValueType,
    pub role: InputRole,
    pub stable_id: Option<u128>,
}

#[derive(Clone, Debug)]
pub struct PlanInputs {
    pub roles: Vec<InputRole>,
    pub stable_ids: Vec<Option<u128>>,
}

pub struct EntrySignature<B> {
    params: Vec<EntryParam>,
    by_value: HashMap<ValueId, usize>,
    _backend: PhantomData<fn() -> B>,
}

impl<B> EntrySignature<B> {
    pub fn new(params: Vec<EntryParam>) -> Self {
        let mut by_value = HashMap::with_capacity(params.len());
        for (idx, param) in params.iter().enumerate() {
            by_value.insert(param.id, idx);
        }
        Self {
            params,
            by_value,
            _backend: PhantomData,
        }
    }

    pub(crate) fn param(&self, value: ValueId) -> Option<&EntryParam> {
        self.by_value
            .get(&value)
            .and_then(|idx| self.params.get(*idx))
    }

    pub(crate) fn role_of(&self, value: ValueId) -> Option<InputRole> {
        self.param(value).map(|p| p.role)
    }

    pub(crate) fn stable_id_of(&self, value: ValueId) -> Option<u128> {
        self.param(value).and_then(|p| p.stable_id)
    }

    pub(crate) fn plan_inputs(&self) -> PlanInputs {
        let mut roles = Vec::with_capacity(self.params.len());
        let mut stable_ids = Vec::with_capacity(self.params.len());
        for param in &self.params {
            roles.push(param.role);
            stable_ids.push(param.stable_id);
        }
        PlanInputs { roles, stable_ids }
    }

    pub(crate) fn get_or_add_param(
        &mut self,
        function: &mut Function,
        role: InputRole,
        stable_id: Option<u128>,
        ty: ValueType,
    ) -> Result<ValueId> {
        if role == InputRole::Param {
            let stable_id =
                stable_id.ok_or_else(|| anyhow!("param inputs must have stable ids"))?;
            if let Some(existing) = self
                .params
                .iter()
                .find(|p| p.role == InputRole::Param && p.stable_id == Some(stable_id))
            {
                if existing.ty != ty {
                    return Err(anyhow!(
                        "derived param stable id {} type mismatch: {:?} vs {:?}",
                        stable_id,
                        existing.ty,
                        ty
                    ));
                }
                return Ok(existing.id);
            }
        }

        let id = ValueId(next_free_value_id(function));
        function.parameter_ids.push(id);
        function.parameters.push(ty.clone());
        let idx = self.params.len();
        self.params.push(EntryParam {
            id,
            ty,
            role,
            stable_id,
        });
        self.by_value.insert(id, idx);
        Ok(id)
    }

    pub(crate) fn remove_params_by_live_set(
        &mut self,
        function: &mut Function,
        live: &std::collections::HashSet<ValueId>,
    ) {
        let mut new_params = Vec::new();
        let mut new_param_ids = Vec::new();
        let mut new_entries = Vec::new();

        for (param_id, param_ty) in function
            .parameter_ids
            .iter()
            .copied()
            .zip(function.parameters.iter().cloned())
        {
            if !live.contains(&param_id) {
                continue;
            }
            new_param_ids.push(param_id);
            new_params.push(param_ty.clone());
            if let Some(entry) = self.param(param_id).cloned() {
                new_entries.push(entry);
            }
        }

        function.parameter_ids = new_param_ids;
        function.parameters = new_params;
        *self = EntrySignature::new(new_entries);
    }
}

fn next_free_value_id(function: &Function) -> u32 {
    let mut max_id = 0u32;
    for id in &function.parameter_ids {
        max_id = max_id.max(id.0);
    }
    for inst in &function.body {
        max_id = max_id.max(inst.id.0);
    }
    max_id.wrapping_add(1)
}
