
source venv/bin/activate

sudo rm -r mavlink

#rm -r pymavlink
#wget https://github.com/ArduPilot/pymavlink/archive/refs/tags/2.4.30.zip
## wget https://github.com/ArduPilot/pymavlink/archive/refs/heads/master.zip
#unzip 2.4.30.zip
#rm 2.4.30.zip
#mv pymavlink-2.4.30 pymavlink

#rm -r mavlink
#wget https://github.com/mavlink/mavlink/archive/refs/tags/1.0.12.zip
#unzip 1.0.12.zip
#rm 1.0.12.zip
#mv mavlink-1.0.12 mavlink

#git clone --depth 1 --branch 1.0.12 https://github.com/mavlink/mavlink.git --recursive
git clone --depth 1 https://github.com/mavlink/mavlink.git --recursive

rm -r -f mavlink/.git*
rm -r -f mavlink/pymavlink/.git*

cd ..
