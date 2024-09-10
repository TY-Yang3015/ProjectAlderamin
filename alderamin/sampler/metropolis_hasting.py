import jax
import jax.numpy as jnp
import jax.random as random
from flax.training.train_state import TrainState
from jax import vmap, pmap, lax
from flax import struct
from functools import partial
from tqdm import tqdm

from alderamin.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem

from einops import rearrange


# TODO: organise import
# TODO: numerical precision control

@struct.dataclass
class WalkerState:
    positions: jnp.ndarray
    step_size: jnp.ndarray
    key: jnp.ndarray

    def propose(self):
        new_key, subkey = random.split(self.key)
        move = random.normal(subkey, self.positions.shape,
                             ) * self.step_size
        proposed_positions = self.positions + move
        return self.replace(key=new_key), proposed_positions


class MetropolisHastingSampler:
    def __init__(self,
                 system: GlobalSystem,
                 batch_size: int,
                 sampling_seed: int,
                 target_acceptance: float,
                 init_width: float,
                 sample_width: float,
                 sample_width_adapt_freq: int,
                 computation_dtype: jnp.dtype | str = "float32"
                 ):
        self.num_of_electrons: int = system.total_electrons
        self.nuc_positions: jnp.ndarray = jnp.array([member.position for member in system.nucleus_list])
        self.spins: jnp.ndarray = jnp.array([electron.spin for electron in system.electrons_list])

        self.batch_size: int = batch_size
        self.num_of_walkers: int = self.batch_size
        self.sampling_key: random.PRNGKey = random.PRNGKey(sampling_seed)
        self.target_acceptance: float = target_acceptance

        self.init_width: float = init_width
        self.sample_width: float = sample_width
        self.sample_width_adapt_freq: int = sample_width_adapt_freq
        self.computation_dtype: jnp.dtype = computation_dtype

        self.electron_nuc_pair: jnp.ndarray = jnp.array(system.electron_to_nucleus)

        self.critical_key: random.PRNGKey = random.PRNGKey(sampling_seed * 42)

        self.walker_state = self.initialise_walkers()

        self.global_memory: jnp.array = jnp.array([])

    def initialise_walkers(self) -> WalkerState:
        init_positions: jnp.ndarray = random.normal(self.sampling_key,
                                                    shape=(self.num_of_walkers, self.num_of_electrons, 3)
                                                    ) * self.init_width

        nuc_pos_array: jnp.ndarray = self.nuc_positions[self.electron_nuc_pair]
        nuc_pos_array = jnp.tile(nuc_pos_array, (self.num_of_walkers, 1, 1))
        init_positions += nuc_pos_array

        self.sampling_key, _ = random.split(self.sampling_key)
        return WalkerState(
            positions=init_positions,
            step_size=jnp.ones((self.num_of_walkers, 1, 1)) * self.sample_width,
            key=random.PRNGKey(*random.randint(self.sampling_key, (1,), -1e5, 1e5)),
        )

    @partial(jax.jit, static_argnums=0)
    def log_multivariate_gaussian(self, x: jnp.ndarray, mu: jnp.ndarray,
                                  sigma: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log probability of x under a multivariate Gaussian distribution.

        Parameters:
        - x: The data points with shape (4096, 7, 3) (jnp array).
        - mu: The mean vector (3-dimensional jnp array).
        - sigma: The covariance matrix (3 x 3 jnp array).

        Returns:
        - log_probs: Log probabilities with shape (4096, 7) (jnp array).
        """
        # Get dimensions for clarity
        batch_size, n_points, dim = x.shape

        # Subtract the mean from each point (broadcasting mu over the first two dimensions)
        diff = x - mu  # Shape: (4096, 7, 3)

        # Calculate the log determinant of the covariance matrix
        sign, logdet = jnp.linalg.slogdet(sigma)

        # Solve for the Mahalanobis distance for each data point
        # TODO: remove inverse for stability
        sigma_inv = jnp.linalg.inv(sigma)
        mahalanobis_term = jnp.einsum('ijk,kl,ijl->ij', diff, sigma_inv, diff)  # Shape: (4096, 7)

        # Compute the log probability for each point
        log_probs = -0.5 * (3 * jnp.log(2 * jnp.pi) + logdet + mahalanobis_term)  # Shape: (4096, 7)

        return log_probs

    def _burn_in_distribution(self, x: jnp.ndarray) -> jnp.ndarray:
        log_probs = jnp.zeros(shape=self.batch_size)
        for nuc_pos in self.nuc_positions:
            log_probs += self.log_multivariate_gaussian(x, nuc_pos,
                                                        jnp.eye(3) * self.init_width).prod(axis=-1)
        return log_probs

    @partial(jax.jit, static_argnums=0)
    def _burn_in_step(self, walker_state: WalkerState) -> tuple[WalkerState, jnp.ndarray]:
        walker_state, proposed_positions = walker_state.propose()

        current_log_prob = self._burn_in_distribution(walker_state.positions)
        proposed_log_prob = self._burn_in_distribution(proposed_positions)

        accept_probs = jnp.exp(proposed_log_prob - current_log_prob)
        accept_probs = jnp.minimum(accept_probs, 1.0)

        walker_state, decisions = self._mh_accept_step(walker_state, accept_probs, proposed_positions)
        return walker_state, jnp.array(decisions, dtype=jnp.int32)

    def _mh_accept_step(self, walker_state: WalkerState,
                        accept_probs: jnp.ndarray,
                        new_positions: jnp.ndarray) -> tuple[WalkerState, jnp.ndarray]:
        accept_decisions = random.uniform(walker_state.key, shape=(walker_state.positions.shape[0], )) < accept_probs
        updated_positions = jnp.where(accept_decisions[:, None, None], new_positions, walker_state.positions)
        return walker_state.replace(positions=updated_positions), accept_decisions

    @partial(jax.jit, static_argnums=0)
    def _adapt_step_size(self, memory: jnp.ndarray,
                         walker_state: WalkerState) -> tuple[jnp.ndarray, WalkerState]:
        accept_rate = jnp.sum(memory) / len(memory)
        new_size = walker_state.step_size * jnp.exp((accept_rate - self.target_acceptance)
                                                    / jnp.sqrt(len(memory)))
        walker_state = walker_state.replace(step_size=new_size)
        memory = jnp.array([])
        return memory, walker_state

    def burn_in(self, num_steps) -> jnp.ndarray:
        memory = jnp.array([])
        for step in tqdm(range(num_steps)):
            self.walker_state, decisions = self._burn_in_step(self.walker_state)
            memory = jnp.append(memory, decisions)
            if step % self.sample_width_adapt_freq == 0:
                memory, self.walker_state = self._adapt_step_size(memory, self.walker_state)
        return self.walker_state.positions

    @partial(jax.jit, static_argnums=0)
    def _psiformer_sample_step(self, walker_state: WalkerState,
                               psiformer_train_state: TrainState) -> tuple[WalkerState, jnp.ndarray]:
        walker_state, proposed_positions = walker_state.propose()

        psiformer_input = self.convert_to_psiformer_input(walker_state.positions)
        proposal_input = self.convert_to_psiformer_input(proposed_positions)
        current_log_prob = jnp.log(psiformer_train_state.apply_fn({"params": psiformer_train_state.params}
                                                                  , *psiformer_input))
        proposed_log_prob = jnp.log(psiformer_train_state.apply_fn({"params": psiformer_train_state.params}
                                                                   , *proposal_input))

        accept_probs = jnp.exp(proposed_log_prob - current_log_prob)
        accept_probs = jnp.minimum(accept_probs, 1.0).squeeze(-1)

        walker_state, decisions = self._mh_accept_step(walker_state, accept_probs, proposed_positions)
        return walker_state, decisions

    @partial(jax.jit, static_argnums=0)
    def convert_to_psiformer_input(self, coordinates: jnp.ndarray) \
            -> (jnp.ndarray, jnp.ndarray):
        """

        :param coordinates: sampled cartesian coordinates, with shape (batch, N, 3)
        :return:
        """
        batch = coordinates.shape[0]
        electron_nuclear_features = jnp.zeros((batch, self.num_of_electrons,
                                               len(self.nuc_positions), 4),
                                              dtype=self.computation_dtype)
        single_electron_features = jnp.zeros((batch, self.num_of_electrons, 4),
                                             dtype=self.computation_dtype)

        single_electron_features = single_electron_features.at[..., :3].set(coordinates)
        single_electron_features = single_electron_features.at[:, :, -1].set(self.spins)

        for i in range(self.num_of_electrons):
            for j in range(len(self.nuc_positions)):
                electron_nuclear_features = electron_nuclear_features.at[:, i, j, :3].set(
                    coordinates[:, i, :] - self.nuc_positions[j, :]
                )
                electron_nuclear_features = electron_nuclear_features.at[:, i, j, -1].set(
                    jnp.linalg.norm(coordinates[:, i, :] - self.nuc_positions[j, :], axis=-1)
                )

        return electron_nuclear_features, single_electron_features

    def sample_psiformer(self, psiformer_train_state: TrainState, sample_step: int) -> jnp.ndarray:
        for _ in range(sample_step):
            self.walker_state, decisions = self._psiformer_sample_step(self.walker_state, psiformer_train_state)
            self.global_memory = jnp.append(self.global_memory, decisions)
            if len(self.global_memory) % self.sample_width_adapt_freq == 0:
                self.global_memory, self.walker_state = self._adapt_step_size(self.global_memory, self.walker_state)
        return self.walker_state.positions


"""
a = AtomicNucleus('Li', (0, 0, 0))
c = ElectronNucleusSystem(system_nucleus=a,
                          num_electrons=3).initialize_system()
b = AtomicNucleus('H', (0, 0, 1))
d = ElectronNucleusSystem(system_nucleus=b,
                          num_electrons=18).initialize_system()

e = GlobalSystem(system_member=[c, d]).initialize_system()

sampler = MetropolisHastingSampler(
    system=e,
    batch_size=4096,
    sampling_seed=1919810,
    target_acceptance=0.6,
    init_width=0.1,
    sample_width=0.001,
    sample_width_adapt_freq=100,
)
result = sampler.burn_in(2000)

print(result.shape)
print(sampler.convert_to_psiformer_input(result)[0].shape)
print(sampler.convert_to_psiformer_input(result)[1].shape)
print(sampler.convert_to_psiformer_input(result)[0][0, 0, 0, :])
print(sampler.convert_to_psiformer_input(result)[1][0, 0, :])
result = rearrange(result, 'b i j -> (b i) j')[:, 1:]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(result[:, 0], result[:, 1], 'r.')
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.2, 1.2)
plt.show()
"""
