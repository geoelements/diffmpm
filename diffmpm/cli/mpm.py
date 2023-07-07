import click

from diffmpm import MPM


@click.command()
@click.option(
    "-f", "--file", "filepath", required=True, type=str, help="Input TOML file"
)
@click.version_option(package_name="diffmpm")
def mpm(filepath):
    solver = MPM(filepath)
    solver.solve()
