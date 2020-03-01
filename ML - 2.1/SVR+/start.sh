cd ～/Documents/ML_Project/ML\ -\ 2.1
# cd ~/Documents/ML_Project/ML\ -\ 2.1
git pull
echo git pull Done

cd SVR+
echo start time: $(date)
python3 main.py > result.log &
echo Done!!!!!!!!!!!
echo Finsh time: $(date)