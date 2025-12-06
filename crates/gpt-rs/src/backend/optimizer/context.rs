use std::collections::HashSet;

use crate::backend::param_resolver::ParamResolver;
use crate::backend::spec::PortableBackend;

use super::entry::EntrySignature;

#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    pub pre_max_iters: usize,
    pub post_max_iters: usize,
    pub fixed_point_inner_max_iters: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            pre_max_iters: 2,
            post_max_iters: 4,
            fixed_point_inner_max_iters: usize::MAX,
        }
    }
}

pub struct OptimizeServices<'a, B: PortableBackend + 'static> {
    /// Resolves and stores Param handles by stable id (includes derived/hoisted Params).
    pub params: Option<&'a dyn ParamResolver<Handle = B::TensorHandle>>,
}

pub struct OptimizeContext<'a, B: PortableBackend + 'static> {
    backend: &'a B,
    services: OptimizeServices<'a, B>,
    pub(super) entry: EntrySignature<B>,
    #[allow(dead_code)]
    pub(super) cfg: OptimizeConfig,
    failed_fold_keys: HashSet<u64>,
}

impl<'a, B: PortableBackend + 'static> OptimizeContext<'a, B> {
    pub fn new(
        backend: &'a B,
        services: OptimizeServices<'a, B>,
        entry: EntrySignature<B>,
        cfg: OptimizeConfig,
    ) -> Self {
        Self {
            backend,
            services,
            entry,
            cfg,
            failed_fold_keys: HashSet::new(),
        }
    }

    pub fn backend(&self) -> &'a B {
        self.backend
    }

    pub fn services(&self) -> &OptimizeServices<'a, B> {
        &self.services
    }

    pub fn services_mut(&mut self) -> &mut OptimizeServices<'a, B> {
        &mut self.services
    }

    pub(crate) fn entry(&self) -> &EntrySignature<B> {
        &self.entry
    }

    pub(crate) fn entry_mut(&mut self) -> &mut EntrySignature<B> {
        &mut self.entry
    }

    pub fn is_failed_fold_key(&self, key: u64) -> bool {
        self.failed_fold_keys.contains(&key)
    }

    pub fn record_failed_fold_key(&mut self, key: u64) {
        self.failed_fold_keys.insert(key);
    }
}
