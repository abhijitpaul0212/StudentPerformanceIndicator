from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file=file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="StudentPerformanceIndicator",
    version="1.0.0",
    author="Abhijit",
    author_email="abhijitpaul0212@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
