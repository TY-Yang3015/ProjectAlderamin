from dataclasses import dataclass, field

from alderamin.data import GlobalSystem


@dataclass
class Hyperparams:
    learning_rate: float | str = 1e-5  # adam: 1e-6
    batch_size: int = 4096
    step: int = 10000
    training_seed = 114514
    gradient_clipping: float = 1e-3  # adam: 1e-5
    log_epsilon = 1e-12

    burn_in_steps: int | None = 100
    sample_steps: int = 10

    mad_clipping_factor: int = 5
    scale_input: bool = False

    save_ckpt: bool = True
    ckpt_freq: int = 1000
    eval_freq: int = 1

    load_ckpt_dir: str | None = None
    load_config: bool = True
    ckpt_step: int | None = None


@dataclass
class SamplerSpec:
    system: GlobalSystem | None = None
    batch_size: int = Hyperparams.batch_size
    sampling_seed: int = 114514
    target_acceptance: float = 0.5
    init_width: float = 0.01
    sample_width: float = 0.02
    sample_width_adapt_freq: int = 100
    computation_dtype: str = "float32"


@dataclass
class PsiFormerSpec:
    num_of_determinants: int = 16
    num_of_blocks: int = 2
    num_heads: int = 4
    qkv_size: int = 64
    use_memory_efficient_attention: bool = False
    group: None | int = None

    computation_dtype: str = "float32"
    param_dtype: str = "float32"

    num_of_electrons: int = 0
    num_of_nucleus: int = 0

    def initialize(self, sampler_spec: SamplerSpec):
        if sampler_spec.system is None:
            raise ValueError(
                "System in SamplerSpec must be initialized before PsiFormerSpec."
            )
        self.num_of_electrons = sampler_spec.system.total_electrons
        self.num_of_nucleus = sampler_spec.system.total_nucleus


@dataclass
class PsiFormerConfig:
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    sampler_spec: SamplerSpec = field(default_factory=SamplerSpec)
    nn_spec: PsiFormerSpec = field(default_factory=PsiFormerSpec)

    def initialize(self):
        self.nn_spec.initialize(self.sampler_spec)
        return self


"""
import hydra

from alderamin.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem

a = AtomicNucleus('Li', (0, 0, 0))
c = ElectronNucleusSystem(system_nucleus=a,
                          num_electrons=3).initialize_system()
b = AtomicNucleus('H', (0, 0, 1))
d = ElectronNucleusSystem(system_nucleus=b,
                          num_electrons=18).initialize_system()

e = GlobalSystem(system_member=[c, d]).initialize_system()
p = PsiFormerConfig()
p.sampler_spec.system = e


@hydra.main(version_base=None)
def execute(config: PsiFormerConfig) -> None:
    config.initialize()
    print(config.sampler_spec.system.summary)


if __name__ == "__main__":
    execute(p)
"""
