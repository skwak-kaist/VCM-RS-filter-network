
## Dicts for SFU-HW sequences
seq_dict = {
    "001"         : ("Pandaset", "001_1920x1080_10"),
    "002"         : ("Pandaset", "002_1920x1080_10"),
    "003"         : ("Pandaset", "003_1920x1080_10"),
    "005"         : ("Pandaset", "005_1920x1080_10"),
    "011"         : ("Pandaset", "011_1920x1080_10"),
    "013"         : ("Pandaset", "013_1920x1080_10"),
    "015"         : ("Pandaset", "015_1920x1080_10"),
    "016"         : ("Pandaset", "016_1920x1080_10"),
    "017"         : ("Pandaset", "017_1920x1080_10"),
    "019"         : ("Pandaset", "019_1920x1080_10"),
    "021"         : ("Pandaset", "021_1920x1080_10"),
    "023"         : ("Pandaset", "023_1920x1080_10"),
    "024"         : ("Pandaset", "024_1920x1080_10"),
    "027"         : ("Pandaset", "027_1920x1080_10"),
    "028"         : ("Pandaset", "028_1920x1080_10"),
    "029"         : ("Pandaset", "029_1920x1080_10"),
    "030"         : ("Pandaset", "030_1920x1080_10"),
    "032"         : ("Pandaset", "032_1920x1080_10"),
    "033"         : ("Pandaset", "033_1920x1080_10"),
    "034"         : ("Pandaset", "034_1920x1080_10"),
    "035"         : ("Pandaset", "035_1920x1080_10"),
    "037"         : ("Pandaset", "037_1920x1080_10"),
    "038"         : ("Pandaset", "038_1920x1080_10"),
    "039"         : ("Pandaset", "039_1920x1080_10"),
    "040"         : ("Pandaset", "040_1920x1080_10"),
    "041"         : ("Pandaset", "041_1920x1080_10"),
    "042"         : ("Pandaset", "042_1920x1080_10"),
    "043"         : ("Pandaset", "043_1920x1080_10"),
    "044"         : ("Pandaset", "044_1920x1080_10"),
    "046"         : ("Pandaset", "046_1920x1080_10"),
    "052"         : ("Pandaset", "052_1920x1080_10"),
    "053"         : ("Pandaset", "053_1920x1080_10"),
    "054"         : ("Pandaset", "054_1920x1080_10"),
    "056"         : ("Pandaset", "056_1920x1080_10"),
    "057"         : ("Pandaset", "057_1920x1080_10"),
    "058"         : ("Pandaset", "058_1920x1080_10"),
    "064"         : ("Pandaset", "064_1920x1080_10"),
    "065"         : ("Pandaset", "065_1920x1080_10"),
    "066"         : ("Pandaset", "066_1920x1080_10"),
    "067"         : ("Pandaset", "067_1920x1080_10"),
    "069"         : ("Pandaset", "069_1920x1080_10"),
    "070"         : ("Pandaset", "070_1920x1080_10"),
    "071"         : ("Pandaset", "071_1920x1080_10"),
    "072"         : ("Pandaset", "072_1920x1080_10"),
    "073"         : ("Pandaset", "073_1920x1080_10"),
    "077"         : ("Pandaset", "077_1920x1080_10"),
    "078"         : ("Pandaset", "078_1920x1080_10"),
    "080"         : ("Pandaset", "080_1920x1080_10"),
    "084"         : ("Pandaset", "084_1920x1080_10"),
    "088"         : ("Pandaset", "088_1920x1080_10"),
    "089"         : ("Pandaset", "089_1920x1080_10"),
    "090"         : ("Pandaset", "090_1920x1080_10"),
    "094"         : ("Pandaset", "094_1920x1080_10"),
    "095"         : ("Pandaset", "095_1920x1080_10"),
    "097"         : ("Pandaset", "097_1920x1080_10"),
    "098"         : ("Pandaset", "098_1920x1080_10"),
    "101"         : ("Pandaset", "101_1920x1080_10"),
    "102"         : ("Pandaset", "102_1920x1080_10"),
    "103"         : ("Pandaset", "103_1920x1080_10"),
    "105"         : ("Pandaset", "105_1920x1080_10"),
    "106"         : ("Pandaset", "106_1920x1080_10"),
    "109"         : ("Pandaset", "109_1920x1080_10"),
    "110"         : ("Pandaset", "110_1920x1080_10"),
    "112"         : ("Pandaset", "112_1920x1080_10"),
    "113"         : ("Pandaset", "113_1920x1080_10"),
    "115"         : ("Pandaset", "115_1920x1080_10"),
    "116"         : ("Pandaset", "116_1920x1080_10"),
    "117"         : ("Pandaset", "117_1920x1080_10"),
    "119"         : ("Pandaset", "119_1920x1080_10"),
    "120"         : ("Pandaset", "120_1920x1080_10"),
    "122"         : ("Pandaset", "122_1920x1080_10"),
    "123"         : ("Pandaset", "123_1920x1080_10"),
    "124"         : ("Pandaset", "124_1920x1080_10"),
    "139"         : ("Pandaset", "139_1920x1080_10"),
    "149"         : ("Pandaset", "149_1920x1080_10"),
    "158"         : ("Pandaset", "158_1920x1080_10"),
}


res_dict = {
    "Pandaset" : (1920, 1080),
}

#fr_dict = { # (IntraPeriod, FrameRate, FramesToBeEncoded, FrameSkip)}
fr_dict = {seq: (32, 10, 80, 0) for seq in seq_dict.keys()}

def __get_qp_list(start_qp, interval, NNIntraQPOffset=-5):
  return [(x, NNIntraQPOffset) for x in range(start_qp, start_qp+interval*6, interval)]

## sequence encoding configuration
#  [(quality, NNIntraQPOffset)]
seq_cfg = {}

seq_cfg["LD_old"] = seq_cfg["LD_e2e"] = seq_cfg["LD_inner"] = seq_cfg["RA_old"] = seq_cfg["RA_e2e"] = seq_cfg["RA_inner"] = {seq: __get_qp_list(22, 5) for seq in seq_dict.keys()}
seq_cfg["AI_old"] = seq_cfg["AI_e2e"] = seq_cfg["AI_inner"] = {seq: __get_qp_list(22, 5) for seq in seq_dict.keys()}

# class definitions
cls_dict = {
  '1': [57, 58, 69, 70, 72, 73, 77],
  '2': [3, 11, 16, 17, 21, 23, 27, 29, 30, 33, 35, 37, 39, 43, 53, 56, 97],
  '3': [88, 89, 90, 95, 109, 112, 113, 115, 117, 119, 122, 124], 
  '4': [64, 65, 66, 67, 71, 78, 149],
  '5': [1, 5, 139, 13, 15, 19, 24, 28, 32, 34, 38, 40, 41, 42, 44, 46, 52, 54],
  '6': [80, 84, 158, 94, 101, 102, 103, 105, 106, 110, 116, 123],
}

seq_roi_cfg_network = "faster_rcnn_X_101_32x8d_FPN_3x"
#print(seq_cfg)
#print(seq_cfg_ai)

