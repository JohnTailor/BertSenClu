from setuptools import setup#,find_packages


base_packages = [
    "numpy>=1.20.0",
    "pandas>=1.1.5",
    "scikit-learn>=0.22.2.post1",
    "sentence-transformers>=0.4.1",
    "streamlit>=1.17.0",
    "pysbd>=0.3.4",
    "pytorch>=1.11.0"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
   name='bertsenclu',
   version='0.1.2',
   author='Johannes Schneider',
   author_email='vollkoff@gmail.com',
   packages=['bertSenClu'],
   #packages=find_packages(exclude=["notebooks", "docs"]),
   url='https://github.com/JohnTailor/BertSenClu',
   project_urls={
         #"Documentation": "https://maartengr.github.io/BERTopic/",
         "Source Code": "https://github.com/JohnTailor/BertSenClu/",
         "Issue Tracker": "https://github.com/JohnTailor/BertSenClu/issues",
     },

   license='LICENSE',
   description='(Bert-)SenClu is a topic modeling technique that leverages sentence transformers to compute topic models.',
   long_description=long_description,
   long_description_content_type="text/markdown",
   install_requires=base_packages,
    keywords="nlp bert topic modeling sentence embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.7',
)
