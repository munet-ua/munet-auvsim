#!/usr/bin/python3

"""
This script calculates basic performance metrics, derived from the Poisonn-app trace-files.
See the trace-file format in the poisson-app code.

The following metrics are calculated:
- Packet Delivery Ratio (PDR)
- Total Network Throughput
- E2E Delay
- Number of corrupted packets
"""

import os
from statistics import mean


# define the trace-files to process
# TRACE_FILES = ["1-0.5-11.tr"]
TRACE_FILES = ["1-0.5-11.tr",  "2-0.5-10.tr"]

# store trace field indices
TRACE_FIELDS = {"TX/RX": 0, "FRAME_SIZE": 5, "DELAY_MS": 6, "CRC_FAIL": 7, "TIMESTAMP": 8}


# init AquaNet, init receive thread, keep sending in main thread
def calcMetrics(traceFiles):
    # combine all trace-files together
    trace = []
    for t in traceFiles:
        if (os.path.exists(t)):
            # strip first two lines, add main contents
            with open(t, "r") as f:
                trace += f.readlines()[2:]
    
    if (len(trace) == 0):
        print("No trace files found. Exiting.")
        return 0

    # read every line, extract and store values
    totalTxCount = 0
    totalRxCount = 0
    totalRxBytes = 0
    totalCorruptCount = 0
    delayList = []
    # to calculate total duration of experiment
    timestampList = []

    for line in trace:
        entry = line.split(":")
        if (entry[TRACE_FIELDS["TX/RX"]] == "t"):
            totalTxCount += 1

        if (entry[TRACE_FIELDS["TX/RX"]] == "r"):
            # do NOT count corrupted packets!
            if (int(entry[TRACE_FIELDS["CRC_FAIL"]]) == 0):
                totalRxCount += 1
                totalRxBytes += int(entry[TRACE_FIELDS["FRAME_SIZE"]])
                delayList.append(float(entry[TRACE_FIELDS["DELAY_MS"]]))
            else:
                totalCorruptCount += 1

        timestampList.append(float(entry[TRACE_FIELDS["TIMESTAMP"]]))

    # calculate and print the metrics
    pdr = round(float(totalRxCount) / totalTxCount, 2)
    throughput = round(8*totalRxBytes/(max(timestampList) - min(timestampList)), 2)
    e2eDelay = round(mean(delayList), 2)

    print("#Nodes\tPDR\tThroughput_bps\tDelay_ms\tCorrupted_pkts")
    print(str(len(TRACE_FILES)) + '\t' + str(pdr) + '\t' + str(throughput) + '\t' + str(e2eDelay) + '\t' + str(totalCorruptCount))


if __name__ == '__main__':
    # run program
    calcMetrics(TRACE_FILES)
