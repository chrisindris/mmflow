{ time tools/dist_test.sh configs/flownet/flownet_thumos14as_test.py checkpoints/flownetc_8x1_sfine_sintel_384x448.pth 2 --out-dir /data/i5O/flownet_out/ --show-dir /data/i5O/flownet_out/ ; } 2>&1 | tee log.txt
