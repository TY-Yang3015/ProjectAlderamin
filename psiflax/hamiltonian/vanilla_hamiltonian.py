import jax.numpy as jnp
import jax
import flax.struct as struct
from flax.training.train_state import TrainState

import psiflax.folx as folx


@struct.dataclass
class VanillaHamiltonian:
    """
    vanilla hamiltonian operator that calculates the sum of kinetic energies and coulomb electric
    potentials.

    :cvar batch_size: int, the batch size.
    :cvar num_of_electrons: int, the number of electrons.
    :cvar nuc_charges: the array of nuclear charges.
    :cvar nuc_positions: the array of nuclear positions.
    """

    batch_size: int
    num_of_electrons: int
    nuc_charges: jnp.ndarray
    nuc_positions: jnp.ndarray

    def coulomb_potential_terms(self, coordinates: jnp.ndarray) -> jnp.ndarray:
        """
        calculate electronic coulomb potentials given a set of coordinates
        and nuclear coordinates.

        :param coordinates: ``jnp.ndarray`` with shape ``(batch, num_of_electrons, 3)``
        :return: ``jnp.ndarray`` with shape ``(batch, 1)``
        """

        elec_elec_term: jnp.ndarray = jnp.zeros((self.batch_size, 1))
        elec_nuc_term: jnp.ndarray = jnp.zeros((self.batch_size, 1))
        nuc_nuc_term: jnp.ndarray = jnp.zeros((self.batch_size, 1))

        for i in range(self.num_of_electrons):
            for j in range(i):
                elec_elec_term = elec_elec_term.at[:, 0].add(
                    1.0
                    / (
                        jnp.linalg.norm(
                            coordinates[:, i, :] - coordinates[:, j, :], axis=-1
                        )
                    )
                )

        for I in range(len(self.nuc_positions)):
            for i in range(self.num_of_electrons):
                elec_nuc_term = elec_nuc_term.at[:, 0].add(
                    (
                        self.nuc_charges[I]
                        / (
                            jnp.linalg.norm(
                                coordinates[:, i, :] - self.nuc_positions[I, :],
                                axis=-1,
                            )
                        )
                    )
                )

        for I in range(len(self.nuc_positions)):
            for J in range(I):
                nuc_nuc_term = nuc_nuc_term.at[:, 0].add(
                    (self.nuc_charges[I] * self.nuc_charges[J])
                    / jnp.linalg.norm(
                        self.nuc_positions[I, :] - self.nuc_positions[J, :], axis=-1
                    )
                )

        return elec_elec_term - elec_nuc_term + nuc_nuc_term

    def get_local_energy(
        self, state: TrainState, batch: jnp.ndarray, complex_out: bool = False
    ) -> jnp.ndarray:
        """
        calculate the local energy of a given state and electron configuration.
        :param state: ``TrainState``, the training state of the neural network ansatz.
        :param batch: ``jnp.ndarray`` with shape ``(batch_size, num_of_electrons, 3)``
        :param complex_out: ``bool``, optional, default ``False``. whether to turn on
                            complex output or not.
        """

        def get_wavefunction(raw_batch):
            wavefunction = state.apply_fn(
                {"params": state.params},
                raw_batch,
            )

            # needed for folx.forward_laplacian
            if wavefunction.shape == (1, 1):
                wavefunction = wavefunction[0][0]

            return wavefunction

        electric_term = self.coulomb_potential_terms(batch)

        laplacian_op = folx.forward_laplacian(get_wavefunction, 0)
        result = jax.vmap(laplacian_op)(batch)
        laplacian, jacobian = result.laplacian, result.jacobian.dense_array
        kinetic_term = -(laplacian + jnp.square(jacobian).sum(-1)) / 2.0

        kinetic_term = kinetic_term.reshape(-1, 1)
        energy_batch = kinetic_term + electric_term

        return energy_batch
