import argparse
import json
from pathlib import Path

from skordinal.experiments import Utilities


def main(general_conf, configurations):
    if not general_conf["basedir"] or not general_conf["datasets"]:
        raise RuntimeError(
            "A dataset has to be defined to run this program.\n"
            + "For more information about using this framework, please refer to the README."
        )

    if not configurations:
        raise RuntimeError(
            "No configuration was defined.\n"
            + "For more information about using this framework, please refer to the README."
        )

    interface = Utilities(general_conf, configurations)
    interface.run_experiment()
    interface.write_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a skordinal experiment.")
    parser.add_argument("config", type=Path, help="Path to a JSON configuration file.")
    args = parser.parse_args()

    with args.config.open() as f:
        recipe = json.load(f)

    main(recipe["general_conf"], recipe["configurations"])
