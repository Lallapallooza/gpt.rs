use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use super::ConversionTarget;

struct ConversionRegistry {
    targets: RwLock<HashMap<String, Arc<dyn ConversionTarget>>>,
}

impl ConversionRegistry {
    fn new() -> Self {
        Self {
            targets: RwLock::new(HashMap::new()),
        }
    }

    fn register(&self, target: Arc<dyn ConversionTarget>) {
        self.targets
            .write()
            .expect("conversion registry poisoned")
            .insert(target.name().to_string(), target);
    }

    fn get(&self, name: &str) -> Option<Arc<dyn ConversionTarget>> {
        self.targets
            .read()
            .expect("conversion registry poisoned")
            .get(name)
            .cloned()
    }

    fn list(&self) -> Vec<String> {
        let mut targets: Vec<String> = self
            .targets
            .read()
            .expect("conversion registry poisoned")
            .keys()
            .cloned()
            .collect();
        targets.sort();
        targets
    }
}

static GLOBAL_REGISTRY: OnceLock<ConversionRegistry> = OnceLock::new();

fn registry() -> &'static ConversionRegistry {
    GLOBAL_REGISTRY.get_or_init(ConversionRegistry::new)
}

pub fn register_target(target: Arc<dyn ConversionTarget>) {
    registry().register(target);
}

pub fn get_target(name: &str) -> Option<Arc<dyn ConversionTarget>> {
    registry().get(name)
}

pub fn list_targets() -> Vec<String> {
    registry().list()
}
