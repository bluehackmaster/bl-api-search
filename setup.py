from setuptools import setup, find_packages
import os
import io

setup(
    name="bl_api_search",
    version="0.0.1",
    description="BlueLens Image Search Request and Response Service",
    # long_description=long_description,

    # The project URL.
    url='https://github.com/BlueLens/bl-api-search',

    # Author details
    author='Youngbok Yoon',
    author_email='master@bluehack.net',

    # Choose your license
    license='MIT',

    classifiers=[
         'Development Status :: 5 - Production/Stable',
         'Intended Audience :: Developers',
         'Natural Language :: English',
         'License :: OSI Approved :: MIT License',
         'Programming Language :: Python',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3.4',
    ],
    packages=find_packages(),
    include_package_data = True, # include files listed in MANIFEST.in
    install_requires=[
        'flask', 'MarkupSafe', 'decorator', 'itsdangerous', 'six', 'brotlipy',
        'raven[flask]', 'flask_limiter', 'Flask-Common',
        'pymongo', 'Pillow', 'numpy', 'boto3', 'tensorflow',
      'stylelens-feature', 'gevent', 'stylelens-search-vector', 'redis'

    ],
)
