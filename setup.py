from setuptools import setup

setup(
    name="difference_gradient_descent",
    version="0.1.0",
    description="Flexible Gradient Descent implementation for problems where gradient of an objective function cannot be found analytically.",
    url="https://github.com/draktr/difference_gradient_descent",
    author="Dragan",
    license="MIT License",
    packages=["difference_gradient_descent"],
    install_requires=["numpy",
                      "pandas",
                      "joblib"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python"
    ],
)
