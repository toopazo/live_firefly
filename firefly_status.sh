
echo "Status of live_firefly --------------------------"
ps aux |grep python
ls -all /home/pi/live_firefly/logs | grep log_

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]
then
    echo "No Python!"
fi