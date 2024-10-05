import hydra
import jax

from psiflax.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from psiflax.trainer import PsiFormerTrainer

from omegaconf import DictConfig


li1 = AtomicNucleus("Li", (-5.051/2, 0., 0.))
li1 = ElectronNucleusSystem(system_nucleus=li1, num_electrons=3).initialize_system()
li2 = AtomicNucleus("Li", (5.051/2, 0., 0.))
li2 = ElectronNucleusSystem(system_nucleus=li2, num_electrons=3).initialize_system()
e = GlobalSystem(system_member=[li1, li2]).initialize_system()
print(e.summary)

@hydra.main(version_base=None, config_path="./config", config_name="base_config")
def execute(config: DictConfig) -> None:
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    jax.config.update("jax_traceback_filtering", 'off')
    print(e.summary)

    trainer = PsiFormerTrainer(config, e)

    pos = trainer.sampler.walker_state.positions
    print(pos.shape)
    import matplotlib.pyplot as plt
    from einops import rearrange
    import numpy as np

    # xy_pos = rearrange(pos, 'b n i -> (b n) i')
    xy_pos = pos[:, 0, :]
    x = xy_pos[:, 0]
    y = xy_pos[:, 1]

    # z = np.clip(z, 0, 10)
    plt.figure(figsize=(12, 12))

    ax = plt.gca()
    ax.scatter(pos[:, 1, 0], pos[:, 1, 1], s=5)
    ax.scatter(pos[:, 0, 0], pos[:, 0, 1], s=5)
    ax.scatter(pos[:, 2, 0], pos[:, 2, 1], s=5)
    ax.scatter(pos[:, 3, 0], pos[:, 3, 1], s=5)
    ax.scatter(pos[:, 4, 0], pos[:, 4, 1], s=5)
    ax.scatter(pos[:, 5, 0], pos[:, 5, 1], s=5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

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
    ax.scatter(pos[:, 4, 0], pos[:, 4, 1], c=z, cmap="viridis", s=5)
    ax.scatter(pos[:, 5, 0], pos[:, 5, 1], c=z, cmap="viridis", s=5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    plt.show()


if __name__ == "__main__":
    execute()
