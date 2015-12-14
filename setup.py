#!/usr/bin/env python

from setuptools import setup, find_packages
from distutils.extension import Extension
from os import path

VERSION = '0.1.1'

ENTRY_POINTS = {
    'orange3.addon': (
        'associate = orangecontrib.associate',
    ),
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/datafusion/widgets/__init__.py
        'Associate = orangecontrib.associate.widgets',
    ),
}

if __name__ == '__main__':
    setup(
        name="Orange3-Associate",
        description="Orange add-on for mining frequent itemsets and association rules.",
        long_description=open(path.join(path.dirname(__file__), 'README.md')).read(),
        version=VERSION,
        author='Bioinformatics Laboratory, FRI UL',
        author_email='contact@orange.biolab.si',
        url='https://github.com/biolab/orange3-associate',
        keywords=(
            'FIM', 'frequent itemset mining',
            'FPM', 'frequent pattern mining',
            'frequent patterns',
            'association rules',
            'apriori',
            'fp-growth',
            'orange3 add-on',
        ),
        packages=find_packages(),
        package_data={
            "orangecontrib.associate.widgets": ["icons/*.svg"],
            "orangecontrib.associate": ["*.pyx"],
        },
        install_requires=[
            'Orange',
        ],
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib'],
        ext_modules=[
            Extension("orangecontrib.associate._fpgrowth",
                      sources=[path.sep.join(("orangecontrib", "associate", "_fpgrowth.cpp"))],
                      extra_compile_args=["-std=c++11", "-O3"],
                      language="c++",)
        ],
        classifiers=[
            'Programming Language :: Python',
            'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
        ],
        zip_safe=False,
    )
