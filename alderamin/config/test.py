import hydra
from omegaconf import DictConfig, OmegaConf

from alderamin.data import AtomicNucleus, ElectronNucleusSystem, GlobalSystem


@hydra.main(config_path=".", version_base=None, config_name="base_config")
def main(cfg: DictConfig) -> None:
    a = AtomicNucleus("H", (0, 0, 0))
    b = ElectronNucleusSystem(system_nucleus=a, num_electrons=1).initialize_system()
    c = AtomicNucleus("H", (1.398, 0, 0))
    d = ElectronNucleusSystem(system_nucleus=c, num_electrons=1).initialize_system()

    e = GlobalSystem(system_member=[b, d]).initialize_system()

    cfg.system = e
    print(OmegaConf.to_yaml(cfg))


main()
