#Script for splitting files into e.g. 15 second, 1 minute, 15 minute etc segments

#!/bin/bash


INPUT_DIR= "[path to fill length files]" #modify as needed
OUTPUT_DIR= "[path for cut up output files]" #modify as needed

SEGMENT=15 #or however long (in seconds) output files are to be
#SEGMENT=900 #for 15 minutes


module load ffmpeg

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.wav; do
    base=$(basename "$f" .wav)

    # Parse filename: SITENAME_YYYYMMDD_HHMMSS
    site=${base%%_*}
    rest=${base#*_}
    date=${rest%%_*}
    time=${rest#*_}

    # 1. Skip the extra PER file date
    if [ "$date" == "20190121" ]; then
        echo "Skipping ignored date (20190121): $base"
        continue
    fi

    start_epoch=$(date -d "${date} ${time:0:2}:${time:2:2}:${time:4:2}" +%s)

    # Perform the split
    ffmpeg -i "$f" -f segment -segment_time $SEGMENT -c copy \
        "$OUTPUT_DIR/${base}_%03d.wav"

    i=0
    for seg in "$OUTPUT_DIR/${base}_"*.wav; do
        
        # 2. Check duration to filter out tiny trailing files (<= 1 second)
        dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$seg")
        
        # Use awk to compare floating point numbers (bash can't do floats natively)
        is_short=$(awk -v dur="$dur" 'BEGIN { print (dur <= 1.0) ? 1 : 0 }')
        
        if [ "$is_short" -eq 1 ]; then
            echo "Deleting trailing short file: $(basename "$seg") (${dur}s)"
            rm "$seg"
            continue
        fi

        offset=$((i * SEGMENT))
        new_epoch=$((start_epoch + offset))
        new_time=$(date -d "@$new_epoch" +"%H%M%S")

        new_name="${site}_${date}_${new_time}.wav"
        mv "$seg" "$OUTPUT_DIR/$new_name"

        ((i++))
    done
done
