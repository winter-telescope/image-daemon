"""
Module for downloading needed data
Adapted from:
     https://github.com/winter-telescope/mirar/blob/main/mirar/downloader/get_test_data.py
     Author: Robert Stein (@robertdstein)
"""

import os
from pathlib import Path

from image_daemon.paths import BASEDIR
from image_daemon.utils.execute_cmd import run_local

DATA_URL = "git@github.com:winter-telescope/winter_data.git"

data_dir = os.path.join(BASEDIR, "data", os.path.basename(DATA_URL.replace(".git", "")))

# Make sure the directory has the proper parents
Path(os.path.dirname(data_dir)).mkdir(parents=True, exist_ok=True)

DATA_TAG = "v0.0.0"


def get_current_tag(data_dir):
    """
    Get the currently checked-out tag in the data directory.
    Returns None if not on a tag or the directory is not a git repo.

    :param data_dir: Path to the data directory.
    :return: The current git tag or None.
    """
    try:
        # Check the current tag in the data directory
        tag = run_local(
            f"git -C {data_dir} describe --tags --abbrev=0", verbose=False
        ).strip()
        return tag
    except Exception:
        return None


def git_lfs_pull(data_dir):
    """
    Pull the actual content for LFS files after cloning or switching branches/tags.

    :param data_dir: Path to the data directory.
    :return: None
    """
    try:
        cmd = f"git -C {data_dir} lfs pull"
        print(f"Fetching LFS-managed files. Executing: {cmd}")
        run_local(cmd)
    except Exception as e:
        print(f"Failed to pull LFS files: {e}")


def update_data():
    """
    Updates the test data by fetching the latest version with git, and then
    checking out the specific tagged version.

    Only downloads data if the current tag doesn't match DATA_TAG.

    :return: None
    """
    current_tag = get_current_tag(data_dir)

    if not os.path.isdir(data_dir):
        # Clone the repository if it doesn't exist
        cmd = f"git clone -b {DATA_TAG} --single-branch {DATA_URL} {data_dir}"
        print(f"No test data found. Downloading. Executing: {cmd}")
        run_local(cmd)
        git_lfs_pull(data_dir)  # Ensure LFS files are pulled after cloning

    elif current_tag != DATA_TAG:
        # If the current tag doesn't match, fetch and checkout the new tag
        print(
            f"Current data tag ({current_tag}) does not match {DATA_TAG}. Updating data..."
        )

        cmds = [
            f"git -C {data_dir} fetch origin refs/tags/{DATA_TAG}:refs/tags/{DATA_TAG}",
            f"git -C {data_dir} checkout tags/{DATA_TAG} -b {DATA_TAG}",
        ]
        for cmd in cmds:
            print(f"Executing: {cmd}")
            run_local(cmd)
        git_lfs_pull(data_dir)  # Ensure LFS files are pulled after checking out new tag
    else:
        print(f"Data is already up-to-date with tag {DATA_TAG}.")


def get_data_dir():
    """
    Get the path to the data directory.

    :return: Path to the data directory.
    """
    update_data()

    return data_dir


if __name__ == "__main__":
    get_data_dir()
