#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:57:51 2023

This is a sandbox for analyzing bias images from the WINTER camera and
deciding automatically whether each sensor is in a well-behaved state.

@author: nlourie
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import astropy.visualization
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from image_daemon.paths import BASEDIR

# Disable LaTeX rendering in matplotlib
matplotlib.rcParams["text.usetex"] = False

# Define the data directory
data_dir = os.path.join(BASEDIR, "data")
print(f"data_dir: {data_dir}")


class WinterImage:
    """
    A class to handle the WINTER camera data and headers.

    Attributes:
        filepath (str): The path of the image file.
        filename (str): The name of the image file.
        comment (str): Additional comments about the image.
        logger (Optional[logging.Logger]): Logger for logging messages.
        verbose (bool): Flag to control verbosity of logs.
        imgs (Dict[str, np.ndarray]): Dictionary holding sub-images keyed by address.
        headers (List[Any]): List of headers for each sub-image.
        header (fits.Header): Top-level FITS header.
        _mef_addr_order (List[str]): Order of MEF addresses.
        _board_id_order (List[int]): Order of board IDs.
        _layer_by_addr (Dict[str, int]): Mapping from address to layer index.
        _layer_by_board_id (Dict[int, int]): Mapping from board ID to layer index.
        _board_id_by_addr (Dict[str, int]): Mapping from address to board ID.
        _addr_by_board_id (Dict[int, str]): Mapping from board ID to address.
        _rowcol_locs (List[tuple[int, int]]): Row and column locations for subplots.
        _rowcol_locs_by_addr (Dict[str, tuple[int, int]]): Mapping from address to subplot locations.
    """

    def __init__(
        self,
        data: Union[str, np.ndarray],
        headers: Optional[Dict[str, Any]] = None,
        comment: str = "",
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the WinterImage object and loads the image or data.

        Args:
            data (Union[str, np.ndarray]): File path to a FITS file or a numpy array of image data.
            headers (Optional[Dict[str, Any]], optional): Dictionary of headers corresponding to the image layers. Defaults to None.
            comment (str, optional): Additional comments about the image. Defaults to "".
            logger (Optional[logging.Logger], optional): Logger instance for logging. Defaults to None.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.

        Raises:
            ValueError: If the input data format is not a string or a numpy array.
        """
        self.filepath: str = ""
        self.filename: str = ""
        self.comment: str = comment
        self.logger: Optional[logging.Logger] = logger
        self.verbose: bool = verbose
        self.imgs: Dict[str, np.ndarray] = {}
        self.headers: List[Any] = []
        self.header: fits.Header = fits.Header()

        self._mef_addr_order: List[str] = ["sa", "sb", "sc", "pa", "pb", "pc"]
        self._board_id_order: List[int] = [2, 6, 5, 1, 3, 4]
        self._layer_by_addr: Dict[str, int] = dict(zip(self._mef_addr_order, range(6)))
        self._layer_by_board_id: Dict[int, int] = dict(
            zip(self._board_id_order, range(6))
        )
        self._board_id_by_addr: Dict[str, int] = dict(
            zip(self._mef_addr_order, self._board_id_order)
        )
        self._addr_by_board_id: Dict[int, str] = {
            v: k for k, v in self._board_id_by_addr.items()
        }
        self._rowcol_locs: List[tuple[int, int]] = [
            (0, 1),
            (1, 1),
            (2, 1),
            (2, 0),
            (1, 0),
            (0, 0),
        ]
        self._rowcol_locs_by_addr: Dict[str, tuple[int, int]] = dict(
            zip(self._mef_addr_order, self._rowcol_locs)
        )

        match data:
            case str():
                self.load_image(data, comment)
            case np.ndarray():
                self.load_data(data, headers_dict=headers)
            case _:
                raise ValueError(
                    f"Input data format {type(data)} not valid. Must be a numpy array of sub-images or a filepath."
                )

    def load_image(self, mef_file_path: str, comment: str = "") -> None:
        """
        Loads the data from a MEF FITS file as a numpy array.

        Args:
            mef_file_path (str): The file path to the MEF FITS file.
            comment (str, optional): Additional comments about the image. Defaults to "".
        """
        self.filepath = mef_file_path
        self.filename = os.path.basename(self.filepath)
        self.comment = comment
        self.imgs = {}
        self.headers = []

        with fits.open(self.filepath) as hdu:
            self.header = hdu[0].header

            for ext in hdu[1:]:
                datasec_str: str = ext.header["DATASEC"][1:-1]
                datasec: np.ndarray = np.array(re.split(r"[,:]", datasec_str)).astype(
                    int
                )
                data: np.ndarray = ext.data[
                    datasec[2] : datasec[3], datasec[0] : datasec[1]
                ]

                addr: Optional[str] = ext.header.get("ADDR", None)
                if addr in self._mef_addr_order:
                    self.imgs[addr] = data
                else:
                    boardid: Optional[int] = ext.header.get("BOARD_ID", None)
                    if boardid in self._board_id_order:
                        addr_mapped: str = self._addr_by_board_id[boardid]
                        self.imgs[addr_mapped] = data

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """
        Logs a message with the specified logging level.

        Args:
            msg (str): The message to log.
            level (int, optional): The logging level. Defaults to logging.INFO.
        """
        formatted_msg: str = f"WinterImage {msg}"

        if self.logger is None:
            print(formatted_msg)
        else:
            self.logger.log(level=level, msg=formatted_msg)

    def load_data(
        self,
        imgs_dict: Dict[str, np.ndarray],
        headers_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Loads image data from a dictionary of images.

        Args:
            imgs_dict (Dict[str, np.ndarray]): Dictionary of images keyed by address.
            headers_dict (Optional[Dict[str, Any]], optional): Dictionary of headers for each image. Defaults to None.
        """
        self.imgs = imgs_dict
        if headers_dict is None:
            self.headers = [""] * len(imgs_dict)
        else:
            self.headers = list(headers_dict.values())

    def plot_mosaic(
        self,
        title: Optional[str] = None,
        cbar: bool = False,
        cmap: Union[str, Dict[str, str]] = "gray",
        norm_by: str = "full",
        post_to_slack: bool = False,  # pylint: disable=unused-argument
        savepath: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Plots a mosaic of sub-images with optional color bar and color map.

        Args:
            title (Optional[str], optional): Title of the plot. Defaults to None.
            cbar (bool, optional): Whether to add a color bar. Defaults to False.
            cmap (Union[str, Dict[str, str]], optional): Colormap or a dictionary of colormaps by address. Defaults to "gray".
            norm_by (str, optional): Normalization method ("full", "sensor", or "chan"). Defaults to "full".
            post_to_slack (bool, optional): Not used, but passed as an argument. Defaults to False.
            savepath (Optional[str], optional): Path to save the plotted mosaic. Defaults to None.
            *args (Any): Additional positional arguments passed to `imshow`.
            **kwargs (Any): Additional keyword arguments passed to `imshow`.
        """
        aspect_ratio: float = 1920 / 1080
        w: float = 3
        h: float = w / aspect_ratio

        fig, axarr = plt.subplots(3, 2, figsize=(4 * h, 2.0 * w))

        # Combine all the data to figure out the full-image normalization
        alldata: np.ndarray = np.concatenate(
            [img.flatten() for img in self.imgs.values()]
        )

        for addr in self._mef_addr_order:
            if addr in self.imgs:
                image: np.ndarray = self.imgs[addr]
                # Rotate starboard images by 180 degrees
                if addr.startswith("s"):
                    image = np.rot90(image, 2)
            else:
                image = np.zeros((1081, 1921))

            rowcol: tuple[int, int] = self._rowcol_locs_by_addr[addr]
            row, col = rowcol

            if norm_by.lower() == "full":
                normdata = alldata
            elif norm_by.lower() in ["sensor", "chan"]:
                normdata = image
            else:
                normdata = alldata

            norm = astropy.visualization.ImageNormalize(
                normdata,
                interval=astropy.visualization.ZScaleInterval(),
                stretch=astropy.visualization.SqrtStretch(),
            )
            ax0 = axarr[row, col]

            if isinstance(cmap, str):
                current_cmap: str = cmap
            elif isinstance(cmap, dict):
                current_cmap = cmap.get(addr, "gray")
            else:
                current_cmap = "gray"

            ax0.text(
                990,
                540,
                addr,
                fontsize=60,
                color="white",
                ha="center",
                va="center",
            )

            individual_plot = ax0.imshow(
                image, origin="lower", cmap=current_cmap, norm=norm, *args, **kwargs
            )
            ax0.set_xlabel("X [pixels]")
            ax0.set_ylabel("Y [pixels]")

            ax0.grid(False)
            ax0.axis("off")

        plt.subplots_adjust(wspace=0.03, hspace=0.01)
        if title is not None:
            plt.suptitle(title)

        if cbar:
            # Add colorbar to the last axis
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(individual_plot, cax=cax, orientation="vertical")

        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

    def get_img(self, chan: Union[str, int], index_by: str = "addr") -> np.ndarray:
        """
        Retrieves a sub-image based on the specified channel and indexing method.

        Args:
            chan (Union[str, int]): The channel identifier, either by address (str) or board ID (int).
            index_by (str, optional): How to index the channel, either by 'addr', 'board_id', or 'layer'. Defaults to "addr".

        Raises:
            ValueError: If the indexing scheme is not one of 'addr', 'board_id', or 'layer'.
            KeyError: If the specified channel does not exist.

        Returns:
            np.ndarray: The requested sub-image.
        """
        index_by_lower: str = index_by.lower()
        addr: Optional[str] = None

        if index_by_lower in ["name", "addr"]:
            if not isinstance(chan, str):
                raise TypeError(
                    f"Expected 'chan' to be a str for index_by='{index_by}'."
                )
            addr = chan
        elif index_by_lower in ["id", "board_id"]:
            if not isinstance(chan, int):
                raise TypeError(
                    f"Expected 'chan' to be an int for index_by='{index_by}'."
                )
            addr = self._addr_by_board_id.get(chan)
            if addr is None:
                raise ValueError(f"Board ID {chan} not found.")
        elif index_by_lower == "layer":
            if not isinstance(chan, int):
                raise TypeError(
                    f"Expected 'chan' to be an int for index_by='{index_by}'."
                )
            if 0 <= chan < len(self._mef_addr_order):
                addr = self._mef_addr_order[chan]
            else:
                raise ValueError(f"Layer {chan} is out of range.")
        else:
            raise ValueError(
                f"index scheme '{index_by}' not valid, must be one of 'addr', 'board_id', or 'layer'."
            )

        try:
            img: np.ndarray = self.imgs[addr]
        except KeyError:
            raise KeyError(f"Image for address '{addr}' not found.")

        return img


def validate_image(
    mef_file_path: str,
    template_path: str,
    addrs: Optional[List[str]] = None,
    comment: str = "",
    plot: bool = True,
    savepath: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compares the specified MEF FITS image to a template and determines the health of each sensor.

    Args:
        mef_file_path (str): Path to the MEF FITS file to validate.
        template_path (str): Path to the template MEF FITS file.
        addrs (Optional[List[str]], optional): List of addresses to validate. If None, all addresses in the test data are validated. Defaults to None.
        comment (str, optional): Additional comments to include in the plot title. Defaults to "".
        plot (bool, optional): Whether to generate and display a plot of the validation results. Defaults to True.
        savepath (Optional[str], optional): Path to save the plotted validation results. If None, the plot is not saved. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary containing validation results, including:
            - Each address with a sub-dictionary containing:
                - "okay" (bool): Whether the sensor is in a good state.
                - "mean" (float): Mean of the deviation from the template.
                - "std" (float): Standard deviation of the deviation from the template.
            - "bad_chans" (List[str]): List of addresses with problematic sensors.
            - "good_chans" (List[str]): List of addresses with well-behaved sensors.
    """
    results: Dict[str, Any] = {}
    cmaps: Dict[str, str] = {}
    bad_chans: List[str] = []
    good_chans: List[str] = []

    # Load the test and template data
    test_data = WinterImage(data=mef_file_path)
    template_data = WinterImage(data=template_path)

    # Retrieve all addresses from the test data
    all_addrs: List[str] = list(test_data.imgs.keys())

    if addrs is None:
        addrs = all_addrs

    # Loop through all addresses and evaluate
    for addr in all_addrs:
        if addr in addrs:
            if addr not in template_data.imgs:
                # Template data missing for this address
                test_data.log(
                    f"Address '{addr}' not found in template data.", logging.WARNING
                )
                results[addr] = {
                    "okay": False,
                    "mean": None,
                    "std": None,
                    "error": "Address not found in template data.",
                }
                bad_chans.append(addr)
                cmaps[addr] = "Reds"
                continue

            template_img = template_data.imgs[addr]
            test_img = test_data.imgs[addr]

            # Avoid division by zero and handle invalid values
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.true_divide(test_img, template_img)
                ratio[~np.isfinite(ratio)] = 0  # Set inf and NaN to 0
                data = np.abs(1 - ratio)

            std: float = np.std(data)
            mean: float = np.mean(data)

            if (std > 0.5) or (mean > 0.1):
                # Image is likely bad
                okay: bool = False
                cmaps[addr] = "Reds"
                bad_chans.append(addr)
            else:
                # Image is good
                okay = True
                cmaps[addr] = "gray"
                good_chans.append(addr)

            results[addr] = {
                "okay": okay,
                "mean": float(mean),
                "std": float(std),
            }
        else:
            # Address not specified for validation; skip
            pass

    # Add summary of bad and good channels
    results.update({"bad_chans": bad_chans, "good_chans": good_chans})

    # Plot the validation results if requested
    if plot:
        if len(bad_chans) == 0:
            suptitle = "No Bad Channels!"
        else:
            suptitle = f"Bad Channel(s): {bad_chans}"
        title = f"{suptitle}\n{test_data.filename}"
        if comment:
            title += f"\n{comment}"
        test_data.plot_mosaic(
            cmap=cmaps, title=title, norm_by="chan", savepath=savepath
        )

    return results


if __name__ == "__main__":
    # Template image
    template_path = os.path.join(data_dir, "test", "master_bias.fits")
    template_im = WinterImage(data=template_path)
    template_im.plot_mosaic()

    # # Partial image (commented out)
    # imgpath = os.path.join(topdir, 'data', 'test', 'test_pa_pb_pc_mef.fits')
    # im = WinterImage(data=imgpath)
    # # im.plot_mosaic()
    # result = validate_image(imgpath, template_path, plot=True)
    # print(result)

    # Full image
    imgpath = os.path.join(data_dir, "test", "test_full_mef.fits")
    im = WinterImage(data=imgpath)
    # im.plot_mosaic()
    result = validate_image(
        mef_file_path=imgpath, template_path=template_path, plot=True
    )
    print(f"bad chans:  {result['bad_chans']}")
    print(f"good chans: {result['good_chans']}")
