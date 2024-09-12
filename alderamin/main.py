import hydra

from alderamin.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from alderamin.config import PsiFormerConfig
from alderamin.trainer import PsiFormerTrainer
from absl import app
import logging

a = AtomicNucleus('H', (0, 0, 0))
c = ElectronNucleusSystem(system_nucleus=a,
                          num_electrons=1).initialize_system()
b = AtomicNucleus('H', (1.423, 0, 0))
d = ElectronNucleusSystem(system_nucleus=b,
                          num_electrons=1).initialize_system()

e = GlobalSystem(system_member=[c, d]).initialize_system()
p = PsiFormerConfig()
p.sampler_spec.system = e
print(e.summary)


@hydra.main(version_base=None)
def execute(config: PsiFormerConfig) -> None:
    import jax
    jax.config.update("jax_debug_nans", True)
    logging.basicConfig(level=logging.INFO)
    config = config.initialize()
    trainer = PsiFormerTrainer(config)
    trainer.train()

    pos = trainer.sampler.walker_state.positions
    print(pos.shape)
    import matplotlib.pyplot as plt
    from einops import rearrange
    pos = rearrange(pos, 'b i j -> (b i) j')[:, 1:]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(pos[:, 0], pos[:, 1], '.', ms=0.5)
    plt.show()



if __name__ == "__main__":
    execute(p)
