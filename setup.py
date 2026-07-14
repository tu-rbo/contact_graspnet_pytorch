from setuptools import find_packages, setup

packages = [p for p in find_packages() if p.startswith('contact_graspnet_pytorch')]

setup(
    name='contact_graspnet_pytorch',
    version='0.1.0+cpf.1',
    author='multiple',
    description='PyTorch implementation of Contact-GraspNet',
    license='NVIDIA Source Code License for Contact-GraspNet',
    license_files=['License.pdf'],
    packages=packages,
    python_requires='>=3.9',
)
