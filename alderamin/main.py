import hydra

from alderamin.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from alderamin.config import PsiFormerConfig
from alderamin.trainer import PsiFormerTrainer
from absl import app
import logging

a = AtomicNucleus("H", (0, 0, 0))
b = ElectronNucleusSystem(system_nucleus=a, num_electrons=1).initialize_system()
c = AtomicNucleus("H", (1.398, 0, 0))
d = ElectronNucleusSystem(system_nucleus=c, num_electrons=1).initialize_system()

e = GlobalSystem(system_member=[b, d]).initialize_system()
p = PsiFormerConfig()
p.sampler_spec.system = e
print(e.summary)


@hydra.main(version_base=None)
def execute(config: PsiFormerConfig) -> None:
    import jax

    jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_enable_x64", True)
    logging.basicConfig(level=logging.INFO)
    config = config.initialize()
    trainer = PsiFormerTrainer(config)

    pos = trainer.sampler.walker_state.positions
    print(pos.shape)
    import matplotlib.pyplot as plt
    from einops import rearrange

    pos = rearrange(pos, "b i j -> (b i) j")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.plot(pos[:, 0], pos[:, 1], ".", ms=1)
    plt.show()

    state = trainer.train()

    pos = trainer.sampler.walker_state.positions
    print(pos.shape)
    import matplotlib.pyplot as plt
    from einops import rearrange
    import numpy as np

    # xy_pos = rearrange(pos, 'b n i -> (b n) i')
    xy_pos = pos[:, 0, :]
    x = xy_pos[:, 0]
    y = xy_pos[:, 1]

    z = np.square((state.apply_fn({"params": state.params}, pos)))
    # z = np.clip(z, 0, 10)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=z, cmap="viridis", s=1)
    plt.colorbar(scatter, label="z values")

    ax = plt.gca()
    ax.scatter(pos[:, 1, 0], pos[:, 1, 1], c=z, cmap="viridis", s=0.1)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-2, 2)

    plt.show()

    print(state.params)


if __name__ == "__main__":
    execute(p)
