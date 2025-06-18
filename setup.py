from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deep-research-agent",
    version="0.1.0",
    author="Deep Research Agent Team",
    author_email="your-email@example.com",
    description="An intelligent research assistant that generates comprehensive research reports with proper citations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ramsoma/deepresearch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deep-research=deep_research_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "deep_research_agent": ["templates/*.jinja2", "prompts/*.py"],
    },
    keywords="research, ai, citations, reports, dspy, gemini",
    project_urls={
        "Bug Reports": "https://github.com/ramsoma/deepresearch/issues",
        "Source": "https://github.com/ramsoma/deepresearch",
        "Documentation": "https://github.com/ramsoma/deepresearch#readme",
    },
) 