from setuptools import setup, find_packages

setup(
    name='primed_vllm_plugin',
    version='0.15.1',
    author="Prannay Kaul, Elvis Nunez, Aditya Chattopadhyay, Evan Becker, Ben Bowman, Luca Zancato",
    author_email="prannayk@amazon.com, elvisnun@amazon.com, achatto@amazon.com, evbecker@amazon.com, bowmaben@amazon.com, zancato@amazon.com",
    url="https://github.com/awslabs/hybrid-model-factory",
    license="Apache-2.0",
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # Find packages in src/
    extras_require={
        "vllm": ["vllm==0.15.1"]
    },
    entry_points={
        'vllm.general_plugins': [
            "register_hybridqwen = primed_vllm.register:register",
        ]
    }
)
