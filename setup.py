import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'numpy',
    'tqdm',
    'mxnet',
    'matplotlib',
    'opencv',
]

setuptools.setup(
    name="multidet", 
    version="0.0.1",
    author="startx",
    author_email="ydl_startx@163.com",
    description="multi detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ydlstartx/MultiDet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)