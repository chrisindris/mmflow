#!/bin/bash
# This script is to run the program one video at a time automatically, and ensure that each folder has only one video.


IFS=$'\n'
set -f

for ARGUMENT in "$@"; do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)

	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"

	export "$KEY"="$VALUE"
done

#TODO: operation control switch so that the user can decide to convert the videos, annotate or both

all_frames_folder="/data/i5O/THUMOS14/actionformer_subset_i3d_frames_all/"
pseudo_frames_folder="$HOME/models/mmflow/tests/data/pseudo_thumos14as/"
output_root="/data/i5O/flownet_out/"


#echo "source_root = $source_root"
#echo "feature_root = $feature_root"
#echo "destination_root = $destination_root"

# -------------------------------------------------

mkdir -p $output_root


for vid in $(find $all_frames_folder -type d -name "*video*" | sort | tail -n 161); do

  # ensure that there is at least 20GB of data left
  if [ $(expr $(df -B1 /data/ | awk 'NR==2 {print $4}') / 1000000000) -gt 20 ]; then

	unset IFS
	set +f

  vid_basename=$(basename $vid)
  echo $vid_basename # current video

  # delete whatever videos are in the pseudo_frames_folder
  find $pseudo_frames_folder -type d -name "*video*" -exec rm -rf {} +

  # copy the current video to pseudo_frames_folder
  cp -r $all_frames_folder/$vid_basename $pseudo_frames_folder


  # prepare the output folders
  split=$(cut -d '_' -f 2 <<<$vid_basename)
  number=$(cut -d '_' -f 3 <<<$vid_basename)

  out_dir="$output_root/out_flow_${split}_${number}"
  show_dir="$output_root/out_flowmap_${split}_${number}"

  echo $out_dir
  echo $show_dir

  mkdir -p $show_dir

  # run the flownet extraction
  time tools/dist_test.sh configs/flownet/flownet_thumos14as_test.py checkpoints/flownetc_8x1_sfine_sintel_384x448.pth 2 --out-dir $out_dir --show-dir $show_dir


  IFS=$'\n'
	set -f

  fi

done

unset IFS
set +f
