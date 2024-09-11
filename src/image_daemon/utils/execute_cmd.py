"""
Module for executing bash commands
Copied from https://github.com/winter-telescope/mirar/blob/main/mirar/utils/execute_cmd.py
Author: Robert Stein (@robertdstein)
"""

import logging
import subprocess
from subprocess import TimeoutExpired

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Error relating to executing bash command"""


class TimeoutExecutionError(Exception):
    """Error relating to timeout when executing bash command"""


DEFAULT_TIMEOUT = 300.0


def run_local(cmd: str, timeout: float = DEFAULT_TIMEOUT, verbose=True):
    """
    Function to run on local machine using subprocess, with error handling.

    After the specified 'cmd' command has been run, any newly-generated files
    will be copied out of the current directory to 'output_dir'

    Parameters
    ----------
    cmd: A string containing the command you want to use to run sextractor.
    An example would be:
        cmd = '/usr/bin/source-extractor image0001.fits -c sex.config'
    timeout: Time to timeout in seconds

    Returns
    -------

    """

    try:
        # Run command

        rval = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            shell=True,
            timeout=timeout,
        )

        msg = "Successfully executed command. "

        if rval.stdout.decode() != "":
            msg += f"Found the following output: {rval.stdout.decode()}"
        logger.debug(msg)
        return rval.stdout.decode()

    except subprocess.CalledProcessError as err:
        msg = (
            f"Execution Error found when running with command: \n \n '{err.cmd}' \n \n"
            f"This yielded a return code of {err.returncode}. "
            f"The following traceback was found: \n {err.stderr.decode()}"
        )
        if verbose:
            logger.error(msg)
        raise ExecutionError(msg) from err

    except TimeoutExpired as err:
        msg = (
            f"Timeout error found when running with command: \n \n '{err.cmd}' \n \n"
            f"The timeout was set to {timeout} seconds. "
            f"The following traceback was found: \n {err.stderr.decode()}"
        )
        if verbose:
            logger.error(msg)
        raise TimeoutExecutionError(msg) from err
