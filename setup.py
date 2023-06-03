from setuptools import setup

with open("README.md", "r") as f:
    desc = f.read()

setup(
    name="findi-descent",
    version="0.1.0",
    description="FinDi: Finite Difference Gradient Descent can optimize any function, including the ones without analytic form, by employing finite difference numerical differentiation within a gradient descent algorithm.",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/draktr/findi-descent",
    author="draktr",
    license="MIT License",
    packages=["findi-descent"],
    python_requires=">=3.8",
    install_requires=["numpy", "pandas", "numba", "joblib", "optschedule"],
    keywords="optimization, gradient-descent, numerical-analysis, numerical-differentiation",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    project_urls={
        "Documentation": "https://findi-descent.readthedocs.io/en/latest/",
        "Issues": "https://github.com/draktr/findi-descent/issues",
    },
)
