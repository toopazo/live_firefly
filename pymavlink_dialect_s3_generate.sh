
source venv/bin/activate

# Based on https://mavlink.io/en/getting_started/generate_libraries.html#mavgen

# Move to live_firefly/mavlink/
cd mavlink

mkdir generated/mavlink/v1.0/ -p
mkdir generated/mavlink/v2.0/ -p

export PYTHONPATH=$(pwd)

python -m pymavlink.tools.mavgen --lang=Python --wire-protocol=1.0 \
  --output=generated/mavlink/v1.0/development message_definitions/v1.0/development.xml

#    Generate the Python MAVLink libraries for your custom dialect.
#    Copy the generated .py MAVLink dialect library file(s) into the appropriate directory of your clone of the
#    mavlink repository:
#        MAVLink 2: pymavlink/dialects/v20
#        MAVLink 1: pymavlink/dialects/v10
#    Open a command prompt and navigate to the pymavlink directory.
#    If needed, uninstall previous versions:
#
#    pip uninstall pymavlink
#
#    Install dependencies if you have not previously installed pymavlink using pip:
#
#    pip install lxml future
#
#    Run the python setup program:
#
#    python setup.py install --user

# Based on https://mavlink.io/en/mavgen_python/

cp generated/mavlink/v1.0/development.py pymavlink/dialects/v10

python -m pip uninstall pymavlink
python -m pip install lxml future

# Move to live_firefly/mavlink/pymavlink
cd pymavlink
# python setup.py install --user
python setup.py install

# Move to live_firefly/mavlink/
cd ..

# Move to live_firefly/
cd ..
