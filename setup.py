import setuptools


with open("Documentation.rst") as f:
    description = f.read()

setuptools.setup(
    author="Jash Shah",
    author_email="shahjash271@gmail.com",
    name="YouGlance",
    license="MIT",
    description=f"Package for analyzing Youtube Videos from searching by relevant entities to analyzing sentiments and clustering different parts of the video according to your liking ",
    version="v0.0.8",
    long_description=description,
    url="https://github.com/Jash271/YouGlance",
    packages=setuptools.find_packages(),
    python_requires=">=3.6.8",
    install_requires=[
        "scikit-learn",
        "sklearn",
        "numpy",
        "pandas",
        "spacy",
        "nltk",
        "youtube_transcript_api",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
    include_package_data=True,
)
