#python live_ars/ars_logger.py logs
#sleep 10
#python live_esc/kde_uas85uvc/kdecan_logger.py logs
#sleep 10
#python firefly_mavcmd_test.py

deactivate

echo "Starting live_ars --------------------------"
cd /home/pi/live_ars
source venv/bin/activate
#python ars_logger.py . &
python ars_logger.py /home/pi/live_firefly/logs &
sleep 10
deactivate
ps -e | grep python

echo "Starting live_esc --------------------------"
cd /home/pi/live_esc
source venv/bin/activate
cd /home/pi/live_esc/kde_uas85uvc
#python kdecan_logger.py . &
python kdecan_logger.py /home/pi/live_firefly/logs &
sleep 10
deactivate
ps -e | grep python

echo "Checking log files --------------------------"
sleep 10
ls -all /home/pi/live_esc/kde_uas85uvc | grep .kdecan
ls -all /home/pi/live_ars | grep .ars

sleep 10
ls -all /home/pi/live_esc/kde_uas85uvc | grep .kdecan
ls -all /home/pi/live_ars | grep .ars

echo "Starting live_firefly --------------------------"
cd /home/pi/live_firefly
sleep 10
python firefly_mavcmd_test.py