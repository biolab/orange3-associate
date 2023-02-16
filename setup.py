#!/usr/bin/env python

from setuptools import setup, find_packages
from distutils.extension import Extension
from os import path

VERSION = '1.2.0'

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

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.associate.widgets:WIDGET_HELP_PATH',),
}

def do_setup(ext_modules):
    setup(
        name="Orange3-Associate",
        description="Orange add-on for mining frequent itemsets and association rules.",
        long_description=open(path.join(path.dirname(__file__), 'README.pypi')).read(),
        long_description_content_type='text/markdown',
        version=VERSION,
        author='Bioinformatics Laboratory, FRI UL',
        author_email='info@biolab.si',
        url='https://github.com/biolab/orange3-associate',
        keywords=(
            'frequent itemset mining',
            'frequent pattern mining',
            'association rules',
            'apriori',
            'fp-growth',
            'frequent patterns',
            'FIM', 'FPM',
            'orange3 add-on',
        ),
        packages=find_packages(),
        package_data={
            "orangecontrib.associate.widgets": ["icons/*.svg"],
            "orangecontrib.associate": ["*.pyx"],
        },
        entry_points=ENTRY_POINTS,
        install_requires=[
            'numpy',
            'scipy',
            'Orange3>=3.33.0'
        ],
        extras_require={
            'test': ['pytest', 'coverage'],
            'doc': ['sphinx', 'recommonmark', 'sphinx_rtd_theme'],
        },
        namespace_packages=['orangecontrib'],
        ext_modules=ext_modules,
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
        include_package_data=True
    )


if __name__ == '__main__':
    ext_modules=[
        Extension("orangecontrib.associate._fpgrowth",
                  sources=[path.sep.join(("orangecontrib", "associate", "_fpgrowth.cpp"))],
                  extra_compile_args=["-std=c++11", "-O3"],
                  language="c++",)
    ]
    try:
        do_setup(ext_modules)
    except:  # fails if no compiler present, e.g. on WinDOS
        import sys
        print('WARNING: Falling back to NOT compiling extension modules. '
              'Performance should suffer. Enjoy.',
              file=sys.stderr)
        do_setup([])
