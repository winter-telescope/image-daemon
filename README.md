# image-daemon
image handling daemon for focus


## Installation
set up a new conda environment: `conda create -n image-daemon python=3.12`

then clone the repository: 

```shell
git clone git@github.com:winter-telescope/image-daemon.git
cd image-daemon
```

this project uses poetry to manage the packages. Set it up this way:

```shell
conda install poetry -c conda-forge`
poetry install
```

After this is set up, install the library in your new conda environment. From inside the `image-daemon` directory:

```shell
python -m pip install -e .
```

where `-e` keeps the installed version up to date if you make edits to the code within the repository.
