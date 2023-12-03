import argparse
import schedule
import time
import json
from datetime import datetime

file = ""
cur_epoch = -999
val_epoch = -999
threshold = 0
high_acc = -999
high_iou = -999
best_epoch = -999
penalty = 0
logs = []


def parseargs():
    parser = argparse.ArgumentParser(description='Read log files every now and then')
    parser.add_argument('--log_file', help='The working directory')
    parser.add_argument('--threshold', help='The directory containing the dataset')

    args = parser.parse_args()
    return args


def read_log():
    global cur_epoch
    global high_iou
    global high_acc
    global penalty
    global best_epoch
    global val_epoch
    global logs

    with open(file) as txt_file:
        for item in txt_file:
            json_object = json.loads(item)
            if 'mode' in json_object:
                if json_object['mode'] == "val" and json_object['epoch'] > val_epoch:
                # if json_object['mode'] == "val":
                    val_epoch = json_object['epoch']
                    if json_object["mIoU"] > high_iou and json_object['mAcc'] > high_acc:
                        high_iou = json_object["mIoU"]
                        high_acc = json_object['mAcc']
                        penalty = 0
                        best_epoch = json_object['epoch']
                    else:
                        penalty = penalty + 1
            if 'epoch' in json_object:
                epoch_value = json_object['epoch']
                if epoch_value > cur_epoch:
                    cur_epoch = epoch_value

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        log_str = (str(current_time) + " Highest EPOCH: " + str(cur_epoch) + " Best EPOCH: " +
                   str(best_epoch) + " mIou and mAcc: " + str(high_iou) + " " + str(high_acc) +
                   " Penalty level: " + str(penalty))

        print(log_str)
        logs.append(log_str)


args = parseargs()

file = args.log_file
threshold = int(args.threshold)

assert isinstance(file, str)
assert isinstance(threshold, int)

read_log()
schedule.every(5).minutes.do(read_log)

while penalty < threshold:
    schedule.run_pending()
    time.sleep(1)

print("END THE RUN - PENALTY REACHED")
temp = file.split("/")[-2]
temp = temp.split(".")[0]
filename = "logs/" + temp + "_logs.txt"
with open(filename, 'w') as tfile:
    tfile.write('\n'.join(logs))

