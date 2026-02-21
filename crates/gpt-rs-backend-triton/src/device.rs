use libloading::Library;

/// Lightweight runtime probe used by the bootstrap backend.
///
/// Full context/stream initialization is introduced in later milestones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaAvailability {
    Available,
    Unavailable,
}

impl CudaAvailability {
    pub fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Debug, Clone)]
pub struct CudaRuntime {
    availability: CudaAvailability,
}

impl CudaRuntime {
    pub fn probe() -> Self {
        Self {
            availability: probe_cuda_driver(),
        }
    }

    pub fn is_available(&self) -> bool {
        self.availability.is_available()
    }

    pub fn global_is_available() -> bool {
        Self::probe().is_available()
    }
}

fn probe_cuda_driver() -> CudaAvailability {
    // Candidate names for CUDA driver library across Linux and Windows.
    let candidates = ["libcuda.so.1", "libcuda.so", "nvcuda.dll"];
    for candidate in candidates {
        // SAFETY: Probing dynamic library presence only. The library handle is
        // dropped immediately and no symbols are invoked in this bootstrap stage.
        if unsafe { Library::new(candidate) }.is_ok() {
            return CudaAvailability::Available;
        }
    }
    CudaAvailability::Unavailable
}
