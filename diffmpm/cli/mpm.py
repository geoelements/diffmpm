import click

from diffmpm.mpm import MPM


@click.command()  # type: ignore
@click.option(
    "-f", "--file", "filepath", required=True, type=str, help="Input TOML file"
)
@click.version_option(package_name="diffmpm")
def mpm(filepath):
    """CLI utility for DiffMPM."""
    solver = MPM(filepath)
    solver.solve()
