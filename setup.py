from setuptools import setup

setup(
    name="dsvm",
    version="0.1",
    description="An exact linear SVM which uses mixed integer programming to determine if data are linearly separable",
    url="https://www.caam.rice.edu/~dtm3/",
    author="David Mildebrath",
    author_email="dtm3@rice.edu",
    license="MIT",
    packages=["dsvm"],
    zip_safe=False,
)
