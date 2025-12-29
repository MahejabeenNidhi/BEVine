#!/bin/bash

# Continue running even if a command fails
set +e

# Create a log file to track versions
LOG_FILE="training_versions_mmcows.log"
echo "Training Run Log - $(date)" > $LOG_FILE
echo "================================" >> $LOG_FILE

# Command 1
echo "Running: segnet_gcef..."
python world_track.py fit -c configs/t_fit.yml -c configs/d_mmcows_train_segnet_gcef.yml -c configs/m_segnet.yml 2>&1 | tee temp_output_mmcows_1.txt
VERSION=$(grep -o 'lightning_logs/version_[0-9]*' temp_output_mmcows_1.txt | head -1 | grep -o '[0-9]*$')
if [ -z "$VERSION" ]; then
    echo "segnet_gcef - Version: FAILED (no version found)" | tee -a $LOG_FILE
else
    echo "segnet_gcef - Version: $VERSION" | tee -a $LOG_FILE
fi
echo ""

# Command 2
echo "Running: segnet_NOgcef..."
python world_track.py fit -c configs/t_fit.yml -c configs/d_mmcows_train_segnet_NOgcef.yml -c configs/m_segnet.yml 2>&1 | tee temp_output_mmcows_2.txt
VERSION=$(grep -o 'lightning_logs/version_[0-9]*' temp_output_mmcows_2.txt | head -1 | grep -o '[0-9]*$')
if [ -z "$VERSION" ]; then
    echo "segnet_NOgcef - Version: FAILED (no version found)" | tee -a $LOG_FILE
else
    echo "segnet_NOgcef - Version: $VERSION" | tee -a $LOG_FILE
fi
echo ""

# Command 3
echo "Running: mvdet_gcef..."
python world_track.py fit -c configs/t_fit.yml -c configs/d_mmcows_train_mvdet_gcef.yml -c configs/m_mvdet.yml 2>&1 | tee temp_output_mmcows_3.txt
VERSION=$(grep -o 'lightning_logs/version_[0-9]*' temp_output_mmcows_3.txt | head -1 | grep -o '[0-9]*$')
if [ -z "$VERSION" ]; then
    echo "mvdet_gcef - Version: FAILED (no version found)" | tee -a $LOG_FILE
else
    echo "mvdet_gcef - Version: $VERSION" | tee -a $LOG_FILE
fi
echo ""

# Command 4
echo "Running: mvdet_NOgcef..."
python world_track.py fit -c configs/t_fit.yml -c configs/d_mmcows_train_mvdet_NOgcef.yml -c configs/m_mvdet.yml 2>&1 | tee temp_output_mmcows_4.txt
VERSION=$(grep -o 'lightning_logs/version_[0-9]*' temp_output_mmcows_4.txt | head -1 | grep -o '[0-9]*$')
if [ -z "$VERSION" ]; then
    echo "mvdet_NOgcef - Version: FAILED (no version found)" | tee -a $LOG_FILE
else
    echo "mvdet_NOgcef - Version: $VERSION" | tee -a $LOG_FILE
fi
echo ""

echo "================================"
echo "All commands completed. Check $LOG_FILE for version numbers."
echo "Temp outputs saved as temp_output_mmcows_1.txt through temp_output_mmcows_4.txt"
cat $LOG_FILE