
# 1) Remove any previous venv
rm -r venv

# 2) Install python3, pip and venv
sudo apt install python3 -y
sudo apt install python3-pip -y
sudo apt install python3-venv -y
python -m ensurepip --upgrade

# 2) Create virtualenv
python3 -m venv venv
# python_ver=$(./create_venv.py --pver); ${python_ver} -m venv venv
# python3.7 -m venv venv
# python3.8 -m venv venv
# python3.9 -m venv venv

# 4) Activate venv
source venv/bin/activate

# 5) Install apt packages
sudo apt-get install can-utils

# 6) Install python libraries
#python -m pip install --upgrade pip
#python -m pip install --upgrade setuptools wheel
#python -m pip install --upgrade twine
python -m pip install scipy
python -m pip install pyserial
python -m pip install mavsdk
python -m pip install aioconsole
python -m pip install pymavlink
python -m pip install can-utils
python -m pip install python-can
python -m pip install net-tools
python -m pip install pyqt5
python -m pip install pandas
python -m pip install matplotlib

# For some reason I need to run this twice
python -m pip install pymavlink

python -m pip install git+https://github.com/toopazo/toopazo_tools.git
python -m pip install git+https://github.com/toopazo/toopazo_ulg.git
python -m pip install git+https://github.com/toopazo/live_esc.git
python -m pip install git+https://github.com/toopazo/live_ars.git


