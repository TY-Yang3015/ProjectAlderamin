# Project PsiFlax

This is a re-implementation project of some neural-network-ansatz-based variational
Monte-Carlo models using the Google
[**Flax**](https://github.com/google/flax) framework, a JAX-based high-performance
deep learning library.

## Quick Start
Use `cd` command to move to the `PsiFlax` directory, and run the following command 
to install `psiflax`. 

```shell
pip install -e .
```

`psiflax` is compatible with any version of python `>=3.10`. `psiflax` uses `hydra`
to manage config files. To run a VMC for a hydrogen atom, execute the
following script:

```python
import hydra

from psiflax.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from psiflax.trainer import PsiFormerTrainer

h1 = AtomicNucleus("H", (0, 0, 0))
h1 = ElectronNucleusSystem(system_nucleus=h1, num_electrons=1).initialize_system()
h2 = AtomicNucleus("H", (1.4011, 0, 0))
h2 = ElectronNucleusSystem(system_nucleus=h2, num_electrons=1).initialize_system()
sys = GlobalSystem(system_member=[h1, h2]).initialize_system()


@hydra.main(version_base=None, config_path="./config", config_name="base_config")
def execute(config):
    trainer = PsiFormerTrainer(config=config,
                               system=sys)
    trainer.train()

if __name__ == "__main__":
    execute()
```

This should yield an energy value around 1.174, which can be used to check
your installation. 

### Reference
- [Implementation of PsiFormer and FermiNet in
raw JAX](https://github.com/google-deepmind/ferminet) by Google _DeepMind_.

- [Haiku-based implementation of PsiFormer and
FermiNet](https://github.com/deepqmc/deepqmc), released as the library _DeepQMC_.

- [Distributed-Shampoo optimiser
by _Google_](https://github.com/google-research/google-research/tree/master/scalable_shampoo)

- [folx: forward laplacian for JAX](https://pypi.org/project/folx/) by N. Gao.
