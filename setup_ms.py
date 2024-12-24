from setuptools import setup,find_packages
from os import path

current_directory = path.abspath(path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_install_requirements():
    requirements_path = path.join(current_directory, "requirements.txt")
    with open(requirements_path, encoding="utf-8") as fp:
        return fp.read().splitlines()


setup(
    name="boat",
    version="0.1.1",
    packages=find_packages(),
    long_description=long_description,
    url="https://github.com/callous-youth/BOAT",
    license="MIT",
    keywords=[
    "Bilevel-optimization",
    "Learning and vision",
    "Python",
    "Deep learning"],
    install_requires=get_install_requirements(),
    classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.9.0',
    author="Yaohua Liu, Xianghao Jiao, Risheng Liu",
    author_email="liuyaohua.918@gmail.com",
    description="A Bilevel Optimization Toolkit in Python for Learning and Vision Tasks",
)


