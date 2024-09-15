from clu import metric_writers
from clu.data import PyTree
import jax.numpy as jnp
import jax


def log_histograms(
    writer: metric_writers.MultiWriter | metric_writers.SummaryWriter,
    params: PyTree,
    grads: PyTree,
    step: int,
):
    for name, param in params.items():
        leaves, structure = jax.tree_util.tree_flatten(param)

        flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        writer.write_histograms(step, {f"{name}/param": flat_array})
    for name, grad_val in grads.items():
        leaves, structure = jax.tree_util.tree_flatten(grad_val)

        flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        writer.write_histograms(step, {f"{name}/grad": flat_array})
