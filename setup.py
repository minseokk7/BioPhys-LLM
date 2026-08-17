from setuptools import setup, find_packages

setup(
    name="biophys-llm",
    version="1.0.0",
    description="A Grand Unified Bio-Physical Optimization Framework for Large Language Models",
    long_description=open("huggingface_release/README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Advanced Agentic AI Research Initiative",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "psutil>=5.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
