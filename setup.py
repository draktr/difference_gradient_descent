from setuptools import setup

with open("README.md", "r") as f:
    desc = f.read()

setup(
    name="fdgd",
    version="0.1.0",
    description="Finite Difference Gradient Descent (FDGD) can solve any function, including the ones without analytic form, by employing finite difference numerical differentiation within a gradient descent algorithm.",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/draktr/fdgd",
    author="draktr",
    license="MIT License",
    packages=["fdgd"],
    python_requires=">=3.6",
    install_requires=["numpy", "pandas", "joblib", "optschedule"],
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
        "Documentation": "https://fdgd.readthedocs.io/en/latest/",
        "Issues": "https://github.com/draktr/fdgd/issues",
    },
)
