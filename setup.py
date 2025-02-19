from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'  # constant to handle -e . in requirements

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements from the requirements.txt file.
    It removes any -e . (editable install) if it exists.
    """
    requirements = []  # Empty list to store requirements
    with open(file_path) as file_obj:
        # Read all lines and strip leading/trailing spaces/newlines
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Clean each line
        
        # Remove '-e .' if it's present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='RohitD',
    author_email='stat.data247@gmail.com',
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=get_requirements('requirements.txt')  # Fetch the requirements from txt file
)
