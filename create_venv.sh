
# Remove any previous venv
rm -r venv

# Install virtualenv
python_ver=$(./create_venv.py); ${python_ver} -m venv venv
# python3.7 -m venv venv
# python3.8 -m venv venv
# python3.9 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies using pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install --upgrade twine

python -m pip install -U scipy
python -m pip install -U pyserial
python -m pip install -U mavsdk
python -m pip install -U pymavlink
# python -m pip install aioconsole
# python -m pip install -U can-utils
# sudo apt-get install can-utils
python -m pip install -U python-can
python -m pip install -U net-tools
python -m pip install -U pyqt5
python -m pip install -U pandas
python -m pip install -U matplotlib
python -m pip install -U toopazo-tools


# For some reason I need to run this twice
python -m pip install -U pymavlink

python -m pip install git+https://github.com/toopazo/toopazo_ulg.git
python -m pip install git+https://github.com/toopazo/live_esc.git
python -m pip install git+https://github.com/toopazo/live_ars.git


