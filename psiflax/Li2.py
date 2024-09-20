import hydra
import jax

from psiflax.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from psiflax.trainer import PsiFormerTrainer

from omegaconf import DictConfig


li1 = AtomicNucleus("Li", (-2.77306, 0., 0.))
li1 = ElectronNucleusSystem(system_nucleus=li1, num_electrons=3).initialize_system()
li2 = AtomicNucleus("Li", (2.77306, 0., 0.))
li2 = ElectronNucleusSystem(system_nucleus=li2, num_electrons=3).initialize_system()
e = GlobalSystem(system_member=[li1, li2]).initialize_system()

@hydra.main(version_base=None, config_path="./config", config_name="base_config")
def execute(config: DictConfig) -> None:
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    jax.config.update("jax_traceback_filtering", 'off')

    trainer = PsiFormerTrainer(config, e)
    trainer.train()


if __name__ == "__main__":
    execute()
