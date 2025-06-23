from enum import Enum
import os
import sys
import argparse 
import logging

from testing import testing

__author__ = "Miguel Salinas <uo34525@uniovi.es>, Alejandro <uo265351@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'
   
def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Machine Learning Model Trainer")
    parser.add_argument(
        "-case-id",
        "--case-id",
        dest="case_id",
        required=True,
        help="Case unique identifier."
    ) 
    parser.add_argument(
        "-case-id-folder",
        "--case-id-folder",
        dest="case_id_folder",
        required=True,
        help="Choose the case id root folder."
    )        
    parser.add_argument(
        "-model-id",
        "--model-id",
        dest="model_id",
        required=True,
        help="Choose the model id."
    )     
    parser.add_argument(
        '-training-percent',
        '--training-percent',
        dest='training_percent',
        type=int,
        default=70,
        help="Training percent"
    )    
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG.",
        action="store_const",
        const=logging.DEBUG,
    )    
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    # create the output case id folder if not exist
    case_id_folder = os.path.join(args.case_id_folder, args.case_id)
    os.makedirs(case_id_folder, exist_ok=True)

    _logger.info("Tester starts here")
    testing.tester(case_id_folder, args.model_id, args.training_percent)
    _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()   