from setuptools import setup, find_packages

setup(
    name="SURNAME_FIRSTNAME_exersices",  # No capital letters! e.g. "moe_yngve_exersices"
    version="0.0.1",
    author="FULL NAME",  # e.g. "Yngve Mardal Moe"
    author_email="your@email.no",
    description="Exersices for INF200",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
)
