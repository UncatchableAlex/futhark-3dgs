from setuptools import setup
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="futhark_3dgs",
    packages=['futhark_3dgs'],
    include_package_data=True,
    package_data={
        "futhark_rasterizer": ["*.fut", "*.pkg"],
    },
)
