#!/bin/bash

# Get the PID of the process
PID=$(ps aux | grep 'nohup python -u glm4_lora_train.py' | awk '{print $2}')

# Check if the PID exists
if [ -z "$PID" ]; then
    echo "No process found"
else
    # Send termination signal
    kill -9 $PID
    echo "Process $PID has been terminated"
fi