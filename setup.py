from setuptools import setup, find_packages

setup(
    name="gr_dsw",
    version="1.0.0",
    description="Generative-Resilient Dual-Stream Watermarking for Image Recovery",
    author="Aryan Kanojia",
    author_email="aryankanojia354@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch", "torchvision", "numpy", "scipy", "PyWavelets", "opencv-python", "scikit-image"
    ],
)