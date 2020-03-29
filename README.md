## A collection of task-specific NLP models

Every model is located in a separate directory and meets the following requirements:
- there is a `requirements.txt` file in the top-level directory containing fixed versions of
every python library required to run and/or train the model
- all user scripts are located in the top-level directory, there is no need for an end-user to open
any nested directories to run or train the model
- there is a `README.md` file containing the following:
  - installation instructions
  - download links for data and/or pre-trained model files if any are available
  - external files structure description for cases when the data is not available
  - usage examples
- python code is documented using [Google Style dosctrings](http://google.github.io/styleguide/pyguide.html#381-docstrings)
