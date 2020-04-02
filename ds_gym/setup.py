from setuptools import setup

setup(name='ds_gym',
      version='0.0.1',
      install_requires=['gym',
                        'cupy-cuda90',
                        'gensim']  # And any other dependencies foo needs
)

