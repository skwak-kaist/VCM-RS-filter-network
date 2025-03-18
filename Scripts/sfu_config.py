## Dicts for SFU-HW sequences

seq_dict = {
    "Traffic"         : ("A", "Traffic_2560x1600_30"),          # 1
    "ParkScene"       : ("B", "ParkScene_1920x1080_24"),        # 2
    "BasketballDrive" : ("B", "BasketballDrive_1920x1080_50"),  # 3
    "BQTerrace"       : ("B", "BQTerrace_1920x1080_60"),        # 4
    "BasketballDrill" : ("C", "BasketballDrill_832x480_50"),    # 5
    "BQMall"          : ("C", "BQMall_832x480_60"),             # 6
    "PartyScene"      : ("C", "PartyScene_832x480_50"),         # 7
    "RaceHorsesC"     : ("C", "RaceHorses_832x480_30"),         # 8
    "BasketballPass"  : ("D", "BasketballPass_416x240_50"),     # 9
    "BQSquare"        : ("D", "BQSquare_416x240_60"),           # 10
    "BlowingBubbles"  : ("D", "BlowingBubbles_416x240_50"),     # 11
    "RaceHorsesD"     : ("D", "RaceHorses_416x240_30"),         # 12
    "Kimono"          : ("O", "Kimono_1920x1080_24"),           # 13
    "Cactus"          : ("O", "Cactus_1920x1080_50"),           # 14
}


res_dict = {
    "A" : (2560, 1600),
    "B" : (1920, 1080),
    "C" : ( 832,  480),
    "D" : ( 416,  240),
    "E" : (1280,  720),
    "O" : (1920, 1080),
}

fr_dict = { # (IntraPeriod, FrameRate, FramesToBeEncoded, FrameSkip)
    "Traffic"         : (32, 30, 33, 117),
    "ParkScene"       : (32, 24, 33, 207),
    "BasketballDrive" : (64, 50, 97, 403),
    "BQTerrace"       : (64, 60, 129, 471),
    "BasketballDrill" : (64, 50, 97, 403),
    "BQMall"          : (64, 60, 129, 471),
    "PartyScene"      : (64, 50, 97, 403),
    "RaceHorsesC"     : (32, 30, 65, 235),
    "BasketballPass"  : (64, 50, 97, 403),
    "BQSquare"        : (64, 60, 129, 471),
    "BlowingBubbles"  : (64, 50, 97, 403),
    "RaceHorsesD"     : (32, 30, 65, 235),
    "Kimono"          : (32, 24, 33, 207),
    "Cactus"          : (64, 50, 97, 403),
}

def __get_qp_list(start_qp, interval, NNIntraQPOffset=-5):
  return [(x, NNIntraQPOffset) for x in range(start_qp, start_qp+interval*6, interval)]

# sequence encoding configuration
#  [(quality, NNIntraQPOffset)]
seq_cfg = {}

# old CTC scenarios - Rennes 146?

seq_cfg["RA_old"] = seq_cfg["LD_old"] = {
  "Traffic"          : __get_qp_list(35, 4),
  "ParkScene"        : __get_qp_list(30, 5),
  "BasketballDrive"  : __get_qp_list(38, 3),
  "BQTerrace"        : __get_qp_list(38, 3),
  "BasketballDrill"  : __get_qp_list(22, 4),
  "BQMall"           : __get_qp_list(27, 5),
  "PartyScene"       : __get_qp_list(32, 4),
  "RaceHorsesC"      : __get_qp_list(27, 4),
  "BasketballPass"   : __get_qp_list(22, 4),
  "BQSquare"         : __get_qp_list(22, 4),
  "BlowingBubbles"   : __get_qp_list(22, 4),
  "RaceHorsesD"      : __get_qp_list(27, 5),
  "Kimono"           : __get_qp_list(32, 5),
  "Cactus"           : __get_qp_list(44, 2),
}

seq_cfg["AI_old"] = {
    "Traffic"         : __get_qp_list(32, 5),
    "ParkScene"       : __get_qp_list(22, 5),
    "BasketballDrive" : __get_qp_list(22, 5),
    "BQTerrace"       : __get_qp_list(22, 5),
    "BasketballDrill" : __get_qp_list(22, 5),
    "BQMall"          : __get_qp_list(22, 5),
    "PartyScene"      : __get_qp_list(22, 5),
    "RaceHorsesC"     : __get_qp_list(22, 5),
    "BasketballPass"  : __get_qp_list(22, 5),
    "BQSquare"        : __get_qp_list(22, 5),
    "BlowingBubbles"  : __get_qp_list(22, 5),
    "RaceHorsesD"     : __get_qp_list(22, 5),
    "Kimono"          : __get_qp_list(32, 5),
    "Cactus"          : __get_qp_list(22, 5),
}

# old CTC scenarios - Sapporo 148, Geneva 149

seq_cfg["RA_inner_old2"] = seq_cfg["LD_inner_old2"] = seq_cfg["RA_e2e_old2"] = seq_cfg["LD_e2e_old2"] = {
  "Traffic"          : __get_qp_list(32, 4),
  "ParkScene"        : __get_qp_list(20, 4),
  "BasketballDrive"  : __get_qp_list(28, 4),
  "BQTerrace"        : __get_qp_list(28, 4),
  "BasketballDrill"  : __get_qp_list(18, 4),
  "BQMall"           : __get_qp_list(26, 4),
  "PartyScene"       : __get_qp_list(22, 4),
  "RaceHorsesC"      : __get_qp_list(20, 4),
  "BasketballPass"   : __get_qp_list(16, 4),
  "BQSquare"         : __get_qp_list(16, 4),
  "BlowingBubbles"   : __get_qp_list(18, 4),
  "RaceHorsesD"      : __get_qp_list(18, 4),
  "Kimono"           : __get_qp_list(26, 4),
  "Cactus"           : __get_qp_list(32, 4),
}

seq_cfg["AI_inner_old2"] = seq_cfg["AI_e2e_old2"] = {
  "Traffic"          : __get_qp_list(28, 4),
  "ParkScene"        : __get_qp_list(18, 4),
  "BasketballDrive"  : __get_qp_list(18, 4),
  "BQTerrace"        : __get_qp_list(24, 4),
  "BasketballDrill"  : __get_qp_list(16, 4),
  "BQMall"           : __get_qp_list(26, 4),
  "PartyScene"       : __get_qp_list(16, 4),
  "RaceHorsesC"      : __get_qp_list(14, 4),
  "BasketballPass"   : __get_qp_list(14, 4),
  "BQSquare"         : __get_qp_list(14, 4),
  "BlowingBubbles"   : __get_qp_list(16, 4),
  "RaceHorsesD"      : __get_qp_list(14, 4),
  "Kimono"           : __get_qp_list(24, 4),
  "Cactus"           : __get_qp_list(20, 4),
}

# current CTC scenarios - after Geneva 149

seq_cfg["RA_e2e"] = seq_cfg["RA_inner"] = seq_cfg["LD_e2e"] = seq_cfg["LD_inner"] = {
  "Traffic"          : __get_qp_list(32, 4),
  "ParkScene"        : __get_qp_list(16, 4),
  "BasketballDrive"  : __get_qp_list(26, 4),
  "BQTerrace"        : __get_qp_list(26, 4),
  "BasketballDrill"  : __get_qp_list(12, 4),
  "BQMall"           : __get_qp_list(22, 4),
  "PartyScene"       : __get_qp_list(18, 4),
  "RaceHorsesC"      : __get_qp_list(14, 4),
  "BasketballPass"   : __get_qp_list(10, 4),
  "BQSquare"         : __get_qp_list(12, 4),
  "BlowingBubbles"   : __get_qp_list(14, 4),
  "RaceHorsesD"      : __get_qp_list(14, 4),
  "Kimono"           : __get_qp_list(22, 4),
  "Cactus"           : __get_qp_list(20, 4),
}

seq_cfg["AI_e2e"] = seq_cfg["AI_inner"] = {
  "Traffic"          : __get_qp_list(26, 4),
  "ParkScene"        : __get_qp_list(16, 4),
  "BasketballDrive"  : __get_qp_list(20, 4),
  "BQTerrace"        : __get_qp_list(22, 4),
  "BasketballDrill"  : __get_qp_list(10, 4),
  "BQMall"           : __get_qp_list(20, 4),
  "PartyScene"       : __get_qp_list(16, 4),
  "RaceHorsesC"      : __get_qp_list(14, 4),
  "BasketballPass"   : __get_qp_list(10, 4),
  "BQSquare"         : __get_qp_list(12, 4),
  "BlowingBubbles"   : __get_qp_list(14, 4),
  "RaceHorsesD"      : __get_qp_list(12, 4),
  "Kimono"           : __get_qp_list(20, 4),
  "Cactus"           : __get_qp_list(22, 4),
}

seq_roi_cfg_network = "faster_rcnn_X_101_32x8d_FPN_3x"
