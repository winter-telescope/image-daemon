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

import astropy.visualization
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from image_daemon.paths import BASEDIR

matplotlib.rcParams["text.usetex"] = False


data_dir = os.path.join(BASEDIR, "data")
print(f"data_dir: {data_dir}")


# params = {'axes.labelsize': 8,
#       'text.fontsize':   6,
#       'legend.fontsize': 7,
#       'xtick.labelsize': 6,
#       'ytick.labelsize': 6,
#       'text.usetex': True,       # <-- There
#       }
# rcParams.update(params)


class WinterImage:
    """
    set up a class to hold the winter data and headers in a easily handleable
    way
    """

    def __init__(self, data, headers=None, comment="", logger=None, verbose=False):
        self.filepath = ""
        self.logger = logger
        self.verbose = verbose
        self.imgs = {}  # sub-images
        self.headers = []
        self.header = []  # top level header
        self._mef_addr_order = ["sa", "sb", "sc", "pa", "pb", "pc"]
        self._board_id_order = [2, 6, 5, 1, 3, 4]
        self._layer_by_addr = dict(zip(self._mef_addr_order, range(6)))
        self._layer_by_board_id = dict(zip(self._board_id_order, range(6)))
        self._board_id_by_addr = dict(zip(self._mef_addr_order, self._board_id_order))

        self._addr_by_board_id = dict((v, k) for k, v in self._board_id_by_addr.items())
        # how should they be ordered in a subplots sense (row, col)
        self._rowcol_locs = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
        self._rowcol_locs_by_addr = dict(zip(self._mef_addr_order, self._rowcol_locs))

        # load up the data
        match data:
            case str():
                # assume it is a filepath:
                self.load_image(data, comment)
            case np.array:
                # assume we're passing it a numpy array of numpy arrays corresponding
                # to layers of the MEF file in the appropriate order
                self.load_data(data, headers_dict=headers)
            case _:
                raise ValueError(
                    f"input data format {type(data)} not valid. must be numpy array of sub-imgs, or a filepath"
                )

    def load_image(self, mef_file_path, comment=""):
        """
        adapted from winter_utils.quick_calibrate_images.get_split_mef_fits_data
        Get the data from a MEF fits file as a numpy array
        :param fits_filename:
        :return:
        """
        self.filepath = mef_file_path
        self.filename = os.path.basename(self.filepath)
        self.comment = comment
        self.imgs = {}
        self.headers = []

        # layer_data = {key: None for key in self._mef_addr_order}

        with fits.open(self.filepath) as hdu:
            self.header = hdu[0].header

            # go through the layers
            # Iterate through extensions to find and store data based on SENPOS or BOARDID
            for ext in hdu[1:]:
                datasec = (
                    np.array(re.split(r"[,:]", ext.header["DATASEC"][1:-1]))
                ).astype(int)
                data = ext.data[datasec[2] : datasec[3], datasec[0] : datasec[1]]

                addr = ext.header.get("ADDR", None)
                # print(addr)
                if addr in self._mef_addr_order:
                    # layer_data[addr] = data
                    self.imgs.update({addr: data})
                else:
                    boardid = ext.header.get("BOARD_ID", None)
                    # print(f'self._addr_by_board_id = {self._addr_by_board_id}')
                    # print(f'boardid = {boardid}')
                    if boardid in self._board_id_order:
                        addr = self._addr_by_board_id[boardid]
                        # layer_data[addr] = data
                        self.imgs.update({addr: data})

        """
        with fits.open(self.filepath) as hdu:
            num_ext = len(hdu)
            self.header = hdu[0].header
            for ext in range(1, num_ext):
                data = hdu[ext].data
                header = hdu[ext].header
                datasec = (
                    np.array(re.split(r"[,:]", header["DATASEC"][1:-1]))
                ).astype(int)
                # flip rows and columns of datasec
                self.imgs.append(
                    data[datasec[2] : datasec[3], datasec[0] : datasec[1]]
                )
                self.headers.append(header)
        
        # convert imgs to numpy array
        self.imgs = np.array(self.imgs)
        """

    def log(self, msg, level=logging.INFO):
        msg = f"WinterImage {msg}"

        if self.logger is None:
            print(msg)
        else:
            self.logger.log(level=level, msg=msg)

    def load_data(self, imgs_dict, headers_dict=None):
        """
        load in a big dictionary that corresponds to the imgs from another winter_image
        """

        self.imgs = imgs_dict
        if headers_dict is None:
            headers_dict = ["" for img in imgs_dict]
        self.headers = headers_dict

    def plot_mosaic(
        self,
        title=None,
        cbar=False,
        cmap="gray",
        norm_by="full",
        post_to_slack=False,  # pylint: disable=unused-argument
        savepath=None,
        *args,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        aspect_ratio = 1920 / 1080
        w = 3
        h = w / aspect_ratio

        fig, axarr = plt.subplots(3, 2, figsize=(4 * h, 2.0 * w))

        # use an internal handle for this to handle an input list
        _cmap = cmap

        # combine all the data to figure out the full-image normalization
        # if norm_by.lower in ["sensor", "chan"]:
        alldata = np.array([])
        for addr in self.imgs:
            alldata = np.append(alldata, self.imgs[addr])

        for addr in self._mef_addr_order:
            if addr in self.imgs:
                image = self.imgs[addr]
                # the starboard images need a flip
                if "s" in addr:
                    image = np.rot90(np.rot90(image))
            else:
                image = np.zeros((1081, 1921))

            rowcol = self._rowcol_locs_by_addr[addr]
            ind = self._layer_by_addr[addr]

            if norm_by.lower in ["full"]:
                normdata = alldata
            elif norm_by.lower in ["sensor", "chan"]:
                normdata = image
            else:
                normdata = alldata

            norm = astropy.visualization.ImageNormalize(
                normdata,
                interval=astropy.visualization.ZScaleInterval(),
                stretch=astropy.visualization.SqrtStretch(),
            )
            row, col = rowcol
            ax0 = axarr[row, col]

            if isinstance(_cmap, str):
                cmap = _cmap
            elif isinstance(_cmap, dict):
                cmap = _cmap.get(addr, "gray")

            ax0.text(
                990,
                540,
                self._mef_addr_order[ind],
                fontsize=60,
                color="white",
                ha="center",
                va="center",
            )

            ax0.imshow(image, origin="lower", cmap=cmap, norm=norm, *args, **kwargs)
            ax0.set_xlabel("X [pixels]")
            ax0.set_ylabel("Y [pixels]")

            ax0.grid(False)
            ax0.axis("off")

        plt.subplots_adjust(wspace=0.03, hspace=0.01)
        if title is not None:
            plt.suptitle(title)

        if cbar:
            # set up the colorbar. this is a pain in this format...
            # source: https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

    def get_img(self, chan, index_by="addr"):
        """
        index_by is how you want to get the sub img, either by addr (eg 'pa')
        or by board_id.

        sensor corresponds to the sensor you want to get, either by address
        or by board id.
        """

        if index_by.lower() in ["name", "addr"]:
            mef_layer = self._layer_by_addr[chan]
        elif index_by.lower() in ["id", "board_id"]:
            mef_layer = self._layer_by_board_id[chan]
        elif index_by.lower() in ["layer"]:
            mef_layer = chan
        else:
            raise ValueError(
                f"index scheme {index_by} not valid, must be one of 'addr' or 'board_id' or 'layer'"
            )

        # header = self.headers[mef_layer]
        img = self.imgs[mef_layer]

        return img  # , header


def validate_image(
    mef_file_path, template_path, addrs=None, comment="", plot=True, savepath=None
):
    """
    compare the image specified to the template images and decide if it is
    in good shape. return a dictionary of the addresses and whether they're
    "okay" or suspicious and a reboot is merited.
    """
    results = {}
    cmaps = {}
    bad_chans = []
    good_chans = []

    # load the data
    test_data = WinterImage(mef_file_path)

    template_data = WinterImage(template_path)

    # this was the old way: cycle through all layers in the template
    # all_addrs = self.template_data._layer_by_addr

    # instead:
    # cycle through all layers in the test data. ignore any offline sensors
    all_addrs = test_data.imgs.keys()

    if addrs is None:
        addrs = all_addrs

    # now loop through all the images and evaluate
    for addr in all_addrs:
        if addr in addrs:
            data = np.abs(1 - (test_data.imgs[addr] / template_data.imgs[addr]))

            std = np.std(data)
            mean = np.average(data)

            if (std > 0.5) or (mean > 0.1):
                # image is likely bad!!
                okay = False
                cmaps.update({addr: "Reds"})
                bad_chans.append(addr)
            else:
                okay = True
                cmaps.update({addr: "gray"})
                good_chans.append(addr)

            results.update(
                {
                    addr: {
                        "okay": okay,
                        "mean": float(mean),
                        "std": float(std),
                    }
                }
            )
        else:
            # cmaps.append("gray")
            pass

    # print(f'cmaps = {cmaps}')

    # make an easy place to grab all the good and bad channels
    results.update({"bad_chans": bad_chans, "good_chans": good_chans})

    # now plot the result
    if plot:
        if len(bad_chans) == 0:
            suptitle = "No Bad Channels!"
        else:
            suptitle = f"Bad Channel(s): {bad_chans}"
        # title= f"\Huge{{{suptitle}}}\n{testdata.filename}"
        title = f"{suptitle}\n{test_data.filename}"
        if comment != "":
            title += f"\n{comment}"
        test_data.plot_mosaic(
            cmap=cmaps, title=title, norm_by="chan", savepath=savepath
        )

    return results


if __name__ == "__main__":
    # template image
    template_path = os.path.join(data_dir, "test", "master_bias.fits")
    template_im = WinterImage(data=template_path)
    template_im.plot_mosaic()

    # # partial image
    # imgpath = os.path.join(topdir, 'data', 'test', 'test_pa_pb_pc_mef.fits')
    # im = WinterImage(data = imgpath)
    # #im.plot_mosaic()
    # result = validate_image(imgpath, template_path, plot = True)
    # print(result)

    # full image
    imgpath = os.path.join(data_dir, "test", "test_full_mef.fits")
    im = WinterImage(data=imgpath)
    # im.plot_mosaic()
    result = validate_image(imgpath, template_path, plot=True)
    print(f"bad chans:  {result['bad_chans']}")
    print(f"good chans: {result['good_chans']}")
