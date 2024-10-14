import jax.numpy as jnp
import jax
from functools import partial
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
    :cvar complex_output: ``bool``. whether to turn on
                            complex output.
    """

    batch_size: int
    num_of_electrons: int
    nuc_charges: jnp.ndarray
    nuc_positions: jnp.ndarray
    complex_output: bool
    mad_clipping_factor: int = 5

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

    @partial(jax.custom_jvp, nondiff_argnums=(0, ))
    def get_local_energy(
            self, state: TrainState, batch: jnp.ndarray
    ) -> jnp.ndarray:
        """
        calculate the local energy of a given state and electron configuration.
        :param state: ``TrainState``, the training state of the neural network ansatz.
        :param batch: ``jnp.ndarray`` with shape ``(batch_size, num_of_electrons, 3)``
        """

        electric_term = self.coulomb_potential_terms(batch)

        if self.complex_output:
            def get_wavefunction(raw_batch):
                wavefunction, _ = state.apply_fn(
                    {"params": state.params},
                    raw_batch,
                )

                # needed for folx.forward_laplacian
                if wavefunction.shape == (1, 1):
                    wavefunction = wavefunction[0][0]

                return wavefunction

            def get_phase(raw_batch):
                _, phase = state.apply_fn(
                    {"params": state.params},
                    raw_batch,
                )

                # needed for folx.forward_laplacian
                if phase.shape == (1, 1):
                    phase = phase[0][0]

                return phase

            amp_laplacian_op = folx.forward_laplacian(get_wavefunction, 0)
            amp_result = jax.vmap(amp_laplacian_op)(batch)
            amp_laplacian, amp_jacobian = amp_result.laplacian, amp_result.jacobian.dense_array

            phase_laplacian_op = folx.forward_laplacian(get_phase, 0)
            phase_result = jax.vmap(phase_laplacian_op)(batch)
            phase_laplacian, phase_jacobian = phase_result.laplacian, phase_result.jacobian.dense_array
            laplacian = amp_laplacian + 1.j * phase_laplacian
            jacobian = amp_jacobian + 1.j * phase_jacobian
            kinetic_term = -(laplacian + jnp.square(jacobian).sum(axis=-1)) / 2.

            kinetic_term = kinetic_term.reshape(-1, 1)
            energy_batch = kinetic_term + electric_term
        else:
            def get_wavefunction(raw_batch):
                wavefunction = state.apply_fn(
                    {"params": state.params},
                    raw_batch,
                )[0]

                # needed for folx.forward_laplacian
                if wavefunction.shape == (1, 1):
                    wavefunction = wavefunction[0][0]

                return wavefunction

            laplacian_op = folx.forward_laplacian(get_wavefunction, 0)
            result = jax.vmap(laplacian_op)(batch)
            laplacian, jacobian = result.laplacian, result.jacobian.dense_array
            kinetic_term = -(laplacian + jnp.square(jacobian).sum(-1)) / 2.0

            kinetic_term = kinetic_term.reshape(-1, 1)
            energy_batch = kinetic_term + electric_term

        return energy_batch

    @get_local_energy.defjvp
    def total_energy_jvp(self, primals, tangents):
        params, data = primals
        energy_batch = self.get_local_energy(self, params, data)

        n = self.mad_clipping_factor
        if self.complex_output:
            real_energy_batch = jnp.real(energy_batch)
            imag_energy_batch = jnp.imag(energy_batch)

            real_mad = jnp.mean(jnp.abs(real_energy_batch - jnp.median(real_energy_batch)))
            imag_mad = jnp.mean(jnp.abs(imag_energy_batch - jnp.median(imag_energy_batch)))

            real_energy_batch = jnp.clip(real_energy_batch,
                                         jnp.median(real_energy_batch) - (n * real_mad),
                                         jnp.median(real_energy_batch) + (n * real_mad))
            imag_energy_batch = jnp.clip(imag_energy_batch,
                                         jnp.median(imag_energy_batch) - (n * imag_mad),
                                         jnp.median(imag_energy_batch) + (n * imag_mad))

            energy_batch = real_energy_batch + 1.j * imag_energy_batch
        else:
            mean_absolute_deviation = jnp.mean(
                jnp.abs(energy_batch - jnp.median(energy_batch))
            )
            energy_batch = jnp.clip(
                energy_batch,
                jnp.median(energy_batch) - (n * mean_absolute_deviation),
                jnp.median(energy_batch) + (n * mean_absolute_deviation),
            )

        def param_to_psi(state_param):
            psi = params.apply_fn({'params': state_param}, data)
            return psi[0]

        psi_primal, psi_tangent = jax.jvp(param_to_psi, (primals[0].params, ), (tangents[0].params, ))
        diff = energy_batch - energy_batch.mean()

        if self.complex_output:
            clipped_el = energy_batch
            term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) +
                     jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
            term2 = jnp.sum(energy_batch * psi_tangent.real)
            primals_out = (energy_batch.real, )
            device_batch_size = jnp.shape(energy_batch)[0]
            tangents_out = ((term1 - 2 * term2).real / device_batch_size, )
        else:
            primals_out = (energy_batch, )
            device_batch_size = jnp.shape(energy_batch)[0]
            tangents_out = ((psi_tangent * diff) / device_batch_size, )

        return primals_out, tangents_out
