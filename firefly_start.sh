#python live_ars/ars_logger.py logs
#sleep 10
#python live_esc/kde_uas85uvc/kdecan_logger.py logs
#sleep 10
#python mavlink_custom.py

 deactivate

 cd /home/pi/live_ars
 source venv/bin/activate
 python ars_logger.py . &
 sleep 10
 deactivate
 ps -e | grep python

 cd /home/pi/live_esc
 source venv/bin/activate
 cd /home/pi/live_esc/kde_uas85uvc
 python kdecan_logger.py . &
 sleep 10
 deactivate
 ps -e | grep python

 cd /home/pi/live_logs
 sleep 10
 ls -all /home/pi/live_esc/kde_uas85uvc
 ls -all /home/pi/live_ars

 sleep 10
 ls -all /home/pi/live_esc/kde_uas85uvc
 ls -all /home/pi/live_ars

 sleep 10
 python mavlink_custom.py