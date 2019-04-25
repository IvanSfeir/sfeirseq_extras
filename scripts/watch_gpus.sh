#!/bin/bash
TIMEOUT=3
SLEEP=300
while true
do
    reset
    printf %"$(tput cols)"s | tr " " "-"
    for server in dvorak0 dvorak1 decore0 decore1 decore2 ceos
    do
        echo "| $server |"
        printf %"$(tput cols)"s | tr " " "-"
        ssh -o LogLevel=QUIET -t $server "
                        timeout $TIMEOUT nvidia-smi | grep % ;
                        printf \%\"\$(tput cols)\"s | tr \" \" \"-\" ;
                        timeout $TIMEOUT nvidia-smi | awk '/Processes:/,0' | grep -v Processes: | grep -v GPU | grep -v ==== | grep -v +--- ;
                        printf \%\"\$(tput cols)\"s | tr \" \" \"-\" ;
                        timeout $TIMEOUT ps -o user,pid,%cpu,%mem,args --pid \$(nvidia-smi | awk '/Processes:/,0' | grep -v Processes: | grep -v GPU | grep -v ==== | grep -v +--- | tr -s ' ' | cut -d ' ' -f 3) 2> /dev/null;
                        " | grep -v "Connection to "
        printf %"$(tput cols)"s | tr " " "-"
    done
    echo "| OAR |"
    printf %"$(tput cols)"s | tr " " "-"
    ssh -o LogLevel=QUIET -t aker "oarstat" | grep -v "Connection to "
    sleep $SLEEP
done

