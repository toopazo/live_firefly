
echo "Stopping python processes --------------------------"

#ps aux |grep python |awk '{print $2}' |xargs kill
ps aux |grep python |awk '{print $2}' |xargs kill

#ps aux |grep python # show all processes which are matching python pattern
##grep -v 'pattern_of_process_you_dont_want_to_kill' - exclude process you don't want to kill
#awk '{print $2}' # show second field of output, it is PID.
#xargs kill - apply kill