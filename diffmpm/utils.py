from jax.tree_util import tree_flatten, tree_unflatten


def _show_example(structured):
    flat, tree = tree_flatten(structured)
    unflattened = tree_unflatten(tree, flat)
    print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")
