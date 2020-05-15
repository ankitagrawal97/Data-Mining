# Install virtual environment
sudo pip3 install virtualenv

# Create virtual environment
virtualenv venv

# Activate virtual environment
source venv/bin/activate

# Install packages
pip install ipykernel
pip install Keras==1.2.2
python3 -m pip install tensorflow==0.12.1
python -c 'import keras; print(keras.__version__)'
python -c 'import tensorflow; print(tensorflow.__version__)'
python3 -m pip install tqdm
python3 -m pip install numpy
python3 -m pip install pandas
python3 -m pip install matplotlib
python3 -m pip install pickle
python -m pip install PIL
pip install image
pip install nltk
pip install h5py

# Check python executables [must point within virtual environment]
which python
which python3

# Install virtual environment kernel
ipython kernel install --user --name=venv

# Run jupyter notebook
jupyter notebook

# Deactivate virtual environment and exit
deactivate
exit
