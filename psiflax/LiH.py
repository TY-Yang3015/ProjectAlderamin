import hydra
import jax

from psiflax.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from psiflax.trainer import PsiFormerTrainer

from omegaconf import DictConfig


li = AtomicNucleus("Li", (0., 0., 0.))
li = ElectronNucleusSystem(system_nucleus=li, num_electrons=3).initialize_system()
h = AtomicNucleus("H", (3.015, 0., 0.))
h = ElectronNucleusSystem(system_nucleus=h, num_electrons=1).initialize_system()
e = GlobalSystem(system_member=[li, h]).initialize_system()

@hydra.main(version_base=None, config_path="./config", config_name="base_config")
def execute(config: DictConfig) -> None:
    print(e.summary)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    jax.config.update("jax_traceback_filtering", 'off')

    trainer = PsiFormerTrainer(config, e)

    pos = trainer.sampler.walker_state.positions
    print(pos.shape)
    import matplotlib.pyplot as plt
    from einops import rearrange

    pos = rearrange(pos, "b i j -> (b i) j")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-3, 3)
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

    z = np.square(np.exp(state.apply_fn({"params": state.params}, pos)))
    # z = np.clip(z, 0, 10)
    plt.figure(figsize=(12, 12))

    ax = plt.gca()
    ax.scatter(pos[:, 1, 0], pos[:, 1, 1], c=z, cmap="viridis", s=5)
    ax.scatter(pos[:, 0, 0], pos[:, 0, 1], c=z, cmap="viridis", s=5)
    ax.scatter(pos[:, 2, 0], pos[:, 2, 1], c=z, cmap="viridis", s=5)
    ax.scatter(pos[:, 3, 0], pos[:, 3, 1], c=z, cmap="viridis", s=5)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-3, 3)

    plt.show()


if __name__ == "__main__":
    execute()
