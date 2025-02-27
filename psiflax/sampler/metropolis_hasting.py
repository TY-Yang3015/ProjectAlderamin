import jax
import jax.numpy as jnp
import jax.random as random
from flax.training.train_state import TrainState
from flax import struct
from einops import rearrange
from functools import partial

from jax import Array
from tqdm import tqdm

from psiflax.data import GlobalSystem


@struct.dataclass
class WalkerState:
    positions: jnp.ndarray
    step_size: jnp.ndarray
    key: jnp.ndarray

    def propose(self):
        new_key, subkey = random.split(self.key)
        move = (
            random.normal(
                subkey,
                self.positions.shape,
            )
            * self.step_size
        )
        proposed_positions = self.positions + move
        return self.replace(key=new_key), proposed_positions


class MetropolisHastingSampler:
    def __init__(
        self,
        system: GlobalSystem,
        batch_size: int,
        sampling_seed: int,
        acceptance_range: list[float, float],
        init_width: float,
        sample_width: float,
        sample_width_adapt_freq: int,
        log_epsilon: float = 1e-12,
        scale_input: bool = True,
        computation_dtype: jnp.dtype | str = "float32",
    ):
        self.num_of_electrons: int = system.total_electrons
        self.nuc_positions: jnp.ndarray = jnp.array(
            [member.position for member in system.nucleus_list], dtype=computation_dtype
        )

        self.batch_size: int = batch_size
        self.num_of_walkers: int = self.batch_size
        self.sampling_key: random.PRNGKey = random.PRNGKey(sampling_seed)
        self.acceptance_range: jnp.ndarray = jnp.array(acceptance_range)  # [min, max]

        self.init_width: float = init_width
        self.sample_width: float = sample_width
        self.sample_width_adapt_freq: int = sample_width_adapt_freq
        self.computation_dtype: jnp.dtype = computation_dtype
        self.log_epsilon: float = log_epsilon
        self.scale_input: bool = scale_input

        self.electron_nuc_pair: jnp.ndarray = jnp.array(
            system.electron_to_nucleus, dtype=jnp.int32
        )

        self.critical_key: random.PRNGKey = random.PRNGKey(sampling_seed * 42)

        self.walker_state = self.initialise_walkers()

        self.global_memory: jnp.array = []

    def initialise_walkers(self) -> WalkerState:
        init_positions: jnp.ndarray = (
            random.normal(
                self.sampling_key,
                shape=(self.num_of_walkers, self.num_of_electrons, 3),
                dtype=self.computation_dtype,
            )
            * self.init_width
        )

        nuc_pos_array: jnp.ndarray = self.nuc_positions[self.electron_nuc_pair]
        nuc_pos_array = jnp.tile(nuc_pos_array, (self.num_of_walkers, 1, 1))
        init_positions += nuc_pos_array

        self.sampling_key, _ = random.split(self.sampling_key)
        return WalkerState(
            positions=init_positions,
            step_size=jnp.ones(
                (self.num_of_walkers, 1, 1), dtype=self.computation_dtype
            )
            * self.sample_width,
            key=random.PRNGKey(*random.randint(self.sampling_key, (1,), -1e5, 1e5)),
        )

    @partial(jax.jit, static_argnums=0)
    def log_multivariate_gaussian(
        self, x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the log probability of x under a multivariate Gaussian distribution.

        Parameters:
        - x: The data points with shape (4096, 7, 3) (jnp array).
        - mu: The mean vector (3-dimensional jnp array).
        - sigma: The covariance matrix (3 x 3 jnp array).

        Returns:
        - log_probs: Log probabilities with shape (4096, 7) (jnp array).
        """
        batch_size, n_points, dim = x.shape

        diff = x - mu  # (batch, electron, dim)

        # calculate the log det of the covariance
        sign, logdet = jnp.linalg.slogdet(sigma)

        # solve for the Mahalanobis distance
        sigma_inv = jnp.linalg.inv(sigma)
        mahalanobis_term = jnp.einsum(
            "ijk,kl,ijl->ij", diff, sigma_inv, diff
        )  # (batch, electrons)

        # compute the log probability
        log_probs = -0.5 * (
            3 * jnp.log(2 * jnp.pi) + logdet + mahalanobis_term
        )  # (batch, electrons)

        return log_probs

    def _burn_in_distribution(self, x: jnp.ndarray) -> jnp.ndarray:
        log_probs = jnp.zeros(shape=self.batch_size, dtype=self.computation_dtype)
        for nuc_pos in self.nuc_positions:
            log_probs += self.log_multivariate_gaussian(
                x, nuc_pos, jnp.eye(3) * self.init_width
            ).prod(axis=-1)
        return log_probs

    @partial(jax.jit, static_argnums=0)
    def _burn_in_step(
        self, walker_state: WalkerState
    ) -> tuple[WalkerState, jnp.ndarray]:
        walker_state, proposed_positions = walker_state.propose()

        current_log_prob = 2.0 * self._burn_in_distribution(walker_state.positions)
        proposed_log_prob = 2.0 * self._burn_in_distribution(proposed_positions)

        accept_probs = proposed_log_prob - current_log_prob
        accept_probs = jnp.minimum(accept_probs, 0.)

        walker_state, decisions = self._mh_accept_step(
            walker_state, accept_probs, proposed_positions
        )
        return walker_state, jnp.array(decisions, dtype=jnp.int32)

    def _mh_accept_step(
        self,
        walker_state: WalkerState,
        accept_probs: jnp.ndarray,
        new_positions: jnp.ndarray,
    ) -> tuple[WalkerState, jnp.ndarray]:
        accept_decisions = (
            jnp.log(random.uniform(
                walker_state.key,
                shape=(walker_state.positions.shape[0],),
                dtype=self.computation_dtype,
            ))
            < accept_probs
        )
        updated_positions = jnp.where(
            accept_decisions[:, None, None], new_positions, walker_state.positions
        )
        return walker_state.replace(positions=updated_positions), accept_decisions

    @partial(jax.jit, static_argnums=0)
    def _adapt_step_size(
        self, memory: jnp.ndarray, walker_state: WalkerState
    ) -> tuple[list, WalkerState]:
        # (steps, walkers) -> (walkers, steps) -> (walkers, )
        accept_rate = jnp.mean(rearrange(memory, "i j -> j i"), axis=-1)
        accept_rate = accept_rate.reshape(accept_rate.shape[0], 1, 1)
        new_size = walker_state.step_size

        new_size /= jnp.where(accept_rate < jnp.min(self.acceptance_range), 1.1, 1.0)
        new_size *= jnp.where(accept_rate > jnp.max(self.acceptance_range), 1.1, 1.0)

        memory = []
        return memory, walker_state.replace(step_size=new_size)

    def burn_in(self, num_steps) -> jnp.ndarray:
        memory = []
        for _ in tqdm(range(num_steps), desc='burn-in'):
            self.walker_state, decisions = self._burn_in_step(self.walker_state)
            memory.append(decisions)
            if len(memory) % self.sample_width_adapt_freq == 0:
                memory, self.walker_state = self._adapt_step_size(
                    jnp.array(memory), self.walker_state
                )
        return self.walker_state.positions

    @partial(jax.jit, static_argnums=0)
    def _psiformer_sample_step(
        self, walker_state: WalkerState, psiformer_train_state: TrainState
    ) -> tuple[WalkerState, jnp.ndarray, jnp.ndarray]:
        walker_state, proposed_positions = walker_state.propose()

        current_log_prob = 2 * (
            (
                psiformer_train_state.apply_fn(
                    {"params": psiformer_train_state.params}, walker_state.positions
                )[0]
            )
        )
        proposed_log_prob = 2 * (
            (
                psiformer_train_state.apply_fn(
                    {"params": psiformer_train_state.params}, proposed_positions
                )[0]
            )
        )
        log_ratio = proposed_log_prob - current_log_prob

        accept_probs = log_ratio  # (batch, 1)

        accept_probs = jnp.minimum(accept_probs, jnp.zeros_like(accept_probs)).squeeze(
            -1
        )

        walker_state, decisions = self._mh_accept_step(
            walker_state, accept_probs, proposed_positions
        )
        return walker_state, decisions, jnp.exp(accept_probs).mean()

    def sample_psiformer(
        self, psiformer_train_state: TrainState, sample_step: int
    ) -> tuple[Array, float]:
        pmean_final = 0.0
        for _ in range(sample_step):
            self.walker_state, decisions, pmean = self._psiformer_sample_step(
                self.walker_state, psiformer_train_state
            )
            pmean_final += pmean
            self.global_memory.append(decisions)
            if (len(self.global_memory) % self.sample_width_adapt_freq == 0) and (len(self.global_memory) > 0):
                self.global_memory, self.walker_state = self._adapt_step_size(
                    jnp.array(self.global_memory), self.walker_state
                )
        return self.walker_state.positions, pmean_final / sample_step


"""
from einops import rearrange
from psiflax.data import AtomicNucleus, ElectronNucleusSystem

a = AtomicNucleus('H', (0, 0, 0))
c = ElectronNucleusSystem(system_nucleus=a,
                          num_electrons=3).initialize_system()
b = AtomicNucleus('H', (0, 0, 1))
d = ElectronNucleusSystem(system_nucleus=b,
                          num_electrons=3).initialize_system()

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
result = sampler.burn_in(20000)

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
#"""
