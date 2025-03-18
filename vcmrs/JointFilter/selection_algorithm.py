from pathlib import Path
import vcmrs

CKPT_DIR = Path(vcmrs.__file__).parent / 'JointFilter/checkpoints'


def rgb_post_lic_multi_quality(qp):
    return CKPT_DIR / 'rgb_post_lic_multi_quality' / 'resnext101'

def rgb_post_vtm_multi_qp_finetuned(qp):
    return CKPT_DIR / 'rgb_post_vtm_multi_qp_finetuned' / 'resnext101'

def yuv_pre(qp):
    return CKPT_DIR / 'yuv_pre' / 'resnet50'

def rgb_best(qp):
    return CKPT_DIR / 'rgb_best'


def yuv_best(qp):
    return CKPT_DIR / 'yuv_best'


def algo1_timm_mse(qp):
    ckpt_paths = [
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse/q3_v8_oiv6_crop_ld1.0_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse/q2_v8_oiv6_crop_ld0.5_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse/q1_v8_oiv6_crop_ld2.0_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
    ]
    if qp < 30:
        ckpt_path = ckpt_paths[0]
    elif qp < 40:
        ckpt_path = ckpt_paths[1]
    else:
        ckpt_path = ckpt_paths[2]
    return ckpt_path


def algo1_timm_mse_pre(qp):
    ckpt_paths = [
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse_pre/q3_v8_oiv6_crop_ld0.5_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse_pre/q2_v8_oiv6_crop_ld1.0_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse_pre/q1_v8_oiv6_crop_ld2.0_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
    ]
    if qp < 30:
        ckpt_path = ckpt_paths[0]
    elif qp < 40:
        ckpt_path = ckpt_paths[1]
    else:
        ckpt_path = ckpt_paths[2]
    return ckpt_path

def algo1_timm_mse_pre_single_q2(qp):
    ckpt_paths = [
        # Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse_pre/q3_v8_oiv6_crop_ld0.5_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
        Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse_pre/q2_v8_oiv6_crop_ld1.0_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
        # Path("../vcmrs/JointFilter/checkpoints/algo1_timm_mse_pre/q1_v8_oiv6_crop_ld2.0_step50k_seed1_lr1e-3_bs32_ls0_mw8").resolve(),
    ]
    # if qp < 30:
    #     ckpt_path = ckpt_paths[0]
    # elif qp < 40:
    #     ckpt_path = ckpt_paths[1]
    # else:
    #     ckpt_path = ckpt_paths[2]
    # return ckpt_path
    return ckpt_paths[0]


def test_0001(qp):
    ckpt_paths = [
        Path("../../filter/logs/train/runs/2024-09-06_15-53-32/checkpoints/020_re").resolve(), # 42
        Path("../../filter/logs/train/runs/2024-09-06_15-53-47/checkpoints/020_re").resolve(), # 39
        Path("../../filter/logs/train/runs/2024-09-06_15-53-53/checkpoints/020_re").resolve(), # 36
        Path("../../filter/logs/train/runs/2024-09-06_15-53-23/checkpoints/020_re").resolve(), # 33
    ]
    if qp >= 41:
        ckpt_path = ckpt_paths[0]
    elif qp >= 38:
        ckpt_path = ckpt_paths[1]
    elif qp >= 35:
        ckpt_path = ckpt_paths[2]
    elif qp >= 32:
        ckpt_path = ckpt_paths[3]
    else:
        ckpt_path = ckpt_paths[-1]
    return ckpt_path

def test_0002(qp):
    ckpt_paths = [
        Path("../../filter/logs/train/runs/2024-09-09_09-06-36/checkpoints/best_re").resolve(),
        Path("../../filter/logs/train/runs/2024-09-09_09-08-10/checkpoints/best_re").resolve(),
        Path("../../filter/logs/train/runs/2024-09-09_09-09-05/checkpoints/best_re").resolve(),
        Path("../../filter/logs/train/runs/2024-09-09_09-09-24/checkpoints/best_re").resolve(),
    ]
    if qp >= 41:
        ckpt_path = ckpt_paths[0]
    elif qp >= 38:
        ckpt_path = ckpt_paths[1]
    elif qp >= 35:
        ckpt_path = ckpt_paths[2]
    elif qp >= 32:
        ckpt_path = ckpt_paths[3]
    else:
        ckpt_path = ckpt_paths[-1]
    return ckpt_path



# ../../filter/logs/train/runs/2024-10-01_07-01-00/checkpoints/best
# ../../filter/logs/train/runs/2024-10-01_07-02-20/checkpoints/best
# ../../filter/logs/train/runs/2024-10-01_07-02-42/checkpoints/best
# ../../filter/logs/train/runs/2024-10-01_07-03-13/checkpoints/best

def test_0003_pre_q1(qp):
    return Path("../../filter/logs/train/runs/2024-10-01_07-01-00/checkpoints/best_re").resolve()

def test_0003_pre_q2(qp): # ld 4.0
    return Path("../../filter/logs/train/runs/2024-10-01_07-02-20/checkpoints/best_re").resolve()

def test_0003_pre_q2_ld38(qp): # ld 3.8, best in rgb pre
    return Path("../../filter/logs/train/runs/2024-10-01_07-08-21/checkpoints/best_re").resolve()

def yuv_test_0001_pre_q2_ld36_lr1e3(qp):
    return Path("../../filter/logs/train/runs/2024-10-02_18-00-23/checkpoints/best_re").resolve()

def yuv_pre_q2_ld36_lr1e4(qp):
    return Path("../../filter/logs/train/runs/2024-10-02_18-01-15/checkpoints/best_re").resolve()

def yuv_pre_q2_ld35_lr5e4(qp):
    return Path("../../filter/logs/train/runs/2024-10-03_07-27-54/checkpoints/best_re").resolve()

def rgb_joint_q2_ld23_lr1e3(qp): # ld 2.3
    return Path("../../filter/logs/train/runs/2024-09-06_18-19-27/checkpoints/008_re").resolve()

def rgb_joint_mse_q2_ld23_lr1e3(qp):
    return Path("../../filter/logs/train/runs/2024-09-06_20-39-28/checkpoints/008_re").resolve()

def rgb_joint_2stage_q2_ld38_ld38_lr1e3(qp):
    return Path("../../filter/logs/train/runs/2024-10-06_15-29-42/checkpoints/best_re").resolve()

def rgb_joint_2stage_q2_ld38_ld38_lr1e3_fixed(qp):
    return Path("../../filter/logs/train/runs/2024-10-07_03-13-05/checkpoints/best_re").resolve()

def yuv_pre_q3_ld28_lr5e4_017(qp):
    return Path("../../filter/logs/train/runs/2024-10-07_11-35-09/checkpoints/017_re").resolve()

def yuv_pre_q3_ld26_lr5e4_017(qp):
    return Path("../../filter/logs/train/runs/2024-10-07_11-35-17/checkpoints/017_re").resolve()

def yuv_pre_q3_ld24_lr5e4_017(qp):
    return Path("../../filter/logs/train/runs/2024-10-07_11-35-26/checkpoints/017_re").resolve()

def yuv_pre_q3_ld22_lr5e4_017(qp):
    return Path("../../filter/logs/train/runs/2024-10-07_11-36-15/checkpoints/017_re").resolve()

def rgb_joint_2stage_q2_ld38_ld38_lr1e3_bdt_aware(qp):
    return Path("../../filter/logs/train/runs/2024-10-08_11-24-03/checkpoints/013_re").resolve()

def rgb_joint_2stage_q2_ld38_ld38_lr1e3_bdt_aware_bdr_007(qp):
    return Path("../../filter/logs/train/runs/2024-10-10_05-33-05/checkpoints/007_re").resolve()

def rgb_joint_2stage_q2_ld38_ld38_lr1e3_bdt_aware_bdr_026(qp):
    return Path("../../filter/logs/train/runs/2024-10-10_05-33-05/checkpoints/026_re").resolve()

def rgb_pre_q2_ld38_lr1e3_bdt_outer_017(qp): 
    return Path("../../filter/logs/train/runs/2024-10-10_15-24-21/checkpoints/008_re").resolve()

def rgb_pre_q2_ld44_lr1e3_bdt_outer_017(qp): 
    return Path("../../filter/logs/train/runs/2024-10-10_15-24-27/checkpoints/008_re").resolve()

def rgb_pre_q2_ld50_lr1e3_bdt_outer_017(qp): 
    return Path("../../filter/logs/train/runs/2024-10-10_15-24-36/checkpoints/008_re").resolve()

def rgb_pre_q2_ld56_lr1e3_bdt_outer_017(qp): 
    return Path("../../filter/logs/train/runs/2024-10-10_15-24-42/checkpoints/008_re").resolve()

def rgb_pre_q2_ld62_lr1e3_bdt_outer_017(qp): 
    return Path("../../filter/logs/train/runs/2024-10-10_15-25-58/checkpoints/008_re").resolve()

def rgb_joint_2stage_full_training_q2_ld38_best(qp):
    return Path("../../filter/logs/train/runs/2024-10-10_04-59-20/checkpoints/best_re").resolve()

def rgb_joint_2stage_bdt_aware_ver3_q2_ld38_best(qp):
    return Path("../../filter/logs/train/runs/2024-10-11_16-02-43/checkpoints/best_re").resolve()

def rgb_joint_2stage_bdt_aware_ver3_q2_ld38_fix_003(qp): # best in rgb joint
    return Path("../../filter/logs/train/runs/2024-10-12_18-04-16/checkpoints/003_re").resolve()

def rgb_joint_2stage_bdt_aware_ver3_q2_ld38_fix_014(qp):
    return Path("../../filter/logs/train/runs/2024-10-12_18-04-16/checkpoints/014_re").resolve()

def yuv_joint_2stage_bdt_aware_ver3_q2_ld35_009(qp):
    return Path("../../filter/logs/train/runs/2024-10-13_09-24-50/checkpoints/009_re").resolve()

def yuv_pre_q2_ld36_lr1e3_002(qp):
    return Path("../../filter/logs/train/runs/2024-10-14_08-19-15/checkpoints/002_re").resolve()

def yuv_pre_q2_ld38_lr1e3_002(qp):
    return Path("../../filter/logs/train/runs/2024-10-14_08-19-48/checkpoints/002_re").resolve()

def yuv_pre_q2_ld40_lr1e3_002(qp):
    return Path("../../filter/logs/train/runs/2024-10-14_08-21-02/checkpoints/002_re").resolve()

def yuv_pre_q2_ld36_lr1e3_008(qp):
    return Path("../../filter/logs/train/runs/2024-10-14_08-19-15/checkpoints/008_re").resolve()

def yuv_pre_q2_ld38_lr1e3_008(qp):
    return Path("../../filter/logs/train/runs/2024-10-14_08-19-48/checkpoints/008_re").resolve()

def yuv_pre_q2_ld40_lr1e3_008(qp):
    return Path("../../filter/logs/train/runs/2024-10-14_08-21-02/checkpoints/008_re").resolve()

def yuv_pre_q2_ld33_lr5e4_004(qp):
    return Path("../../filter/logs/train/runs/2024-10-15_02-26-26/checkpoints/004_re").resolve()

def yuv_pre_q2_ld34_lr5e4_013(qp):
    return Path("../../filter/logs/train/runs/2024-10-15_02-27-00/checkpoints/013_re").resolve()

def yuv_pre_q2_ld35_lr5e4_010(qp): # best in yuv pre
    return Path("../../filter/logs/train/runs/2024-10-15_02-26-39/checkpoints/010_re").resolve()

def yuv_pre_q2_ld36_lr5e4_006(qp):
    return Path("../../filter/logs/train/runs/2024-10-15_02-26-16/checkpoints/006_re").resolve()

def yuv_pre_q2_ld35_lr5e4_026(qp):
    return Path("../../filter/logs/train/runs/2024-10-15_02-26-39/checkpoints/026_re").resolve()

def yuv_pre_q2_ld35_post_lr5e4_024(qp):
    return Path("../../filter/logs/train/runs/2024-10-16_05-59-50/checkpoints/024_re").resolve()

def yuv_pre_q2_ld35_post_lr1e4_022(qp):
    return Path("../../filter/logs/train/runs/2024-10-16_06-07-39/checkpoints/022_re").resolve()

def yuv_pre_q2_ld35_post_lr1e3_002(qp):
    return Path("../../filter/logs/train/runs/2024-10-16_06-57-13/checkpoints/002_re").resolve()

def hybrid_yuv_pre_rgb_post(qp):
    return Path("../vcmrs/JointFilter/checkpoints/hybrid").resolve()

def yuv_pre_q2_ld25_post_mse_per_lr2e3_003_re(qp): # q2
    return Path("../../filter/logs/train/runs/2024-10-28_08-18-17/checkpoints/003_re").resolve()

def yuv_pre_q2_ld25_post_qvar_mse_per_lr2e3_012(qp): # q-multi
    return Path("../../filter/logs/train/runs/2024-10-29_10-12-29/checkpoints/012_re").resolve()

def yuv_pre_q2_ld25_post_q1_mse_per_lr2e3_007(qp): # q1
    return Path("../../filter/logs/train/runs/2024-10-29_10-19-08/checkpoints/007_re").resolve()

def yuv_pre_q2_ld25_post_q3_mse_per_lr2e3_007(qp): # q3
    return Path("../../filter/logs/train/runs/2024-10-29_10-19-14/checkpoints/007_re").resolve()

def yuv_pre_q2_ld25_post_q4_mse_per_lr2e3_003(qp): # q4
    return Path("../../filter/logs/train/runs/2024-10-29_10-19-20/checkpoints/003_re").resolve()

def yuv_pre_q2_ld25_post_q5_mse_per_lr2e3_002(qp): # q5
    return Path("../../filter/logs/train/runs/2024-10-29_10-19-26/checkpoints/002_re").resolve()

def yuv_pre_q2_ld25_post_q6_mse_per_lr2e3_008(qp): # q6
    return Path("../../filter/logs/train/runs/2024-10-29_10-19-33/checkpoints/008_re").resolve()

def yuv_pre_q2_ld25_post_q7_mse_per_lr2e3_003(qp): # q7
    return Path("../../filter/logs/train/runs/2024-10-29_10-19-39/checkpoints/003_re").resolve()

def yuv_pre_q2_ld25_post_q8_mse_per_lr2e3_013(qp): # q8
    return Path("../../filter/logs/train/runs/2024-10-29_10-20-46/checkpoints/013_re").resolve()

def rgb_joint(qp):
    return CKPT_DIR / 'rgb_joint'

def rgb_pre(qp):
    return CKPT_DIR / 'rgb_joint'

def rgb_joint_oiv6(qp):
    if qp == 33:
        return Path("../../../filter/logs/train/runs/2024-11-22_19-04-59-197913/checkpoints/039_re").resolve()
    elif qp == 36:
        return Path("../../../filter/logs/train/runs/2024-11-22_19-04-41-175980/checkpoints/039_re").resolve()
    elif qp == 39:
        return Path("../../../filter/logs/train/runs/2024-11-22_19-02-56-407023/checkpoints/039_re").resolve()
    elif qp == 42:
        return Path("../../../filter/logs/train/runs/2024-11-22_19-02-51-420639/checkpoints/039_re").resolve()
    else:
        raise ValueError(f"qp {qp} not supported")
    
def rgb_joint_q2_ld25_lr1e3_cosine_pre_mse_004(qp): ###### 25.01 기고, pre
    return Path("../../filter/logs/train/runs/2024-11-25_00-30-49-412570/checkpoints/004_re").resolve()

def rgb_bdt3_joint_q2_ld25_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-11-25_00-30-51-811099/checkpoints/004_re").resolve()

def rgb_joint_q2_ld38_lr1e3_cosine_pre_perc_004(qp):
    return Path("../../filter/logs/train/runs/2024-11-25_10-58-29-146204/checkpoints/004_re").resolve()

def rgb_joint_q2_ld05_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-04_13-07-37-254135/checkpoints/004_re").resolve()

def rgb_joint_q2_ld10_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-04_13-10-18-127498/checkpoints/004_re").resolve()

def rgb_joint_q2_ld15_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-04_13-10-36-585112/checkpoints/004_re").resolve()

def rgb_joint_q2_ld20_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-04_13-10-37-884176/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-04_13-08-57-542013/checkpoints/004_re").resolve()

def rgb_joint_q2_ld30_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_07-56-43-109453/checkpoints/004_re").resolve()

def rgb_joint_q2_ld35_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_07-55-24-707460/checkpoints/004_re").resolve()

def rgb_joint_q2_ld40_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_07-55-29-276352/checkpoints/004_re").resolve()

def rgb_joint_q2_ld45_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_07-55-34-409410/checkpoints/004_re").resolve()

def rgb_joint_q2_ld50_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_07-55-40-328448/checkpoints/004_re").resolve()

def rgb_joint_q2_ld60_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-35-45-216392/checkpoints/004_re").resolve()

def rgb_joint_q2_ld70_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-35-50-383554/checkpoints/004_re").resolve()

def rgb_joint_q2_ld80_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-35-53-256606/checkpoints/004_re").resolve()

def rgb_joint_q2_ld90_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-37-40-696528/checkpoints/004_re").resolve()

def rgb_joint_q2_ld100_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-36-01-389539/checkpoints/004_re").resolve()

def rgb_joint_q2_ld110_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-36-05-292398/checkpoints/004_re").resolve()

def rgb_joint_q2_ld120_lr1e3_cosine_pre_mse_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-08_13-36-08-688227/checkpoints/004_re").resolve()

def yuv_joint_q2_ld25_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_08-04-12-875110/checkpoints/004_re").resolve()

def yuv_joint_q2_ld20_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-07_08-02-38-363824/checkpoints/004_re").resolve()

def rgb_joint_q4_ld06_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-19_07-16-21-736311/checkpoints/004_re").resolve()

def rgb_joint_q6_ld02_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-19_07-17-04-311252/checkpoints/004_re").resolve()

def rgb_joint_q8_ld005_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-22_13-59-25-983528/checkpoints/004_re").resolve()

def rgb_joint_q8_ld002_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-22_13-59-43-519563/checkpoints/004_re").resolve()

def rgb_joint_q1_ld35_lr1e3_cosine_pre_mse_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-22_14-01-32-851207/checkpoints/004_re").resolve()

def rgb_post_q2_ld0_lr1e3_multistep_004(qp):
    return Path("../../filter/logs/train/runs/2024-11-22_12-23-33-923062/checkpoints/004_re").resolve()

def rgb_post_q4_ld0_lr1e3_multistep_004(qp):
    return Path("../../filter/logs/train/runs/2024-11-22_12-24-08-635556/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_pre_mse_004_pre_sep_post_bdt1(qp):
    return Path("../../filter/logs/train/runs/2024-12-29_13-49-56-755047/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_pre_mse_004_pre_sep_post_bdt3(qp):
    return Path("../../filter/logs/train/runs/2024-12-29_13-50-10-656869/checkpoints/004_re").resolve()

def rgb_post_q8_ld0_lr1e3_cosine_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-29_18-29-20-636094/checkpoints/004_re").resolve()

def rgb_post_q6_ld0_lr1e3_cosine_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-29_18-29-15-970905/checkpoints/004_re").resolve()

def rgb_post_q4_ld0_lr1e3_cosine_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-29_18-29-30-563275/checkpoints/004_re").resolve()

def rgb_post_q2_ld0_lr1e3_cosine_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-29_18-29-35-475065/checkpoints/004_re").resolve()

def rgb_post_q8_ld0_lr1e3_cosine_bdt3_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-01-807271/checkpoints/004_re").resolve()

def rgb_post_q6_ld0_lr1e3_cosine_bdt3_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-42-47-932436/checkpoints/004_re").resolve()

def rgb_post_q4_ld0_lr1e3_cosine_bdt3_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-13-710704/checkpoints/004_re").resolve()

def rgb_post_q2_ld0_lr1e3_cosine_bdt3_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-18-417441/checkpoints/004_re").resolve()

def rgb_post_q8_ld0_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-26-684831/checkpoints/004_re").resolve()

def rgb_post_q6_ld0_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-31-145190/checkpoints/004_re").resolve()

def rgb_post_q4_ld0_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-35-082280/checkpoints/004_re").resolve()

def rgb_post_q2_ld0_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2024-12-30_15-43-41-792479/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_q2_post_sep_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_06-43-25-167915/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_q4_post_sep_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_06-45-42-221050/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_q6_post_sep_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_06-43-32-679215/checkpoints/004_re").resolve()

def rgb_joint_q2_ld25_lr1e3_cosine_q8_post_sep_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_06-43-37-294117/checkpoints/004_re").resolve()

def rgb_joint_q2_ld50_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-45-51-321014/checkpoints/004_re").resolve()

def rgb_joint_q4_ld25_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-45-53-694764/checkpoints/004_re").resolve()

def rgb_joint_q6_ld12_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-45-55-785417/checkpoints/004_re").resolve()

def rgb_joint_q8_ld06_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-45-58-051916/checkpoints/004_re").resolve()

def rgb_joint_q2_ld40_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-46-03-040163/checkpoints/004_re").resolve()

def rgb_joint_q4_ld20_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-46-08-595710/checkpoints/004_re").resolve()

def rgb_joint_q6_ld08_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-46-12-977821/checkpoints/004_re").resolve()

def rgb_joint_q8_ld03_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-01_17-46-17-053988/checkpoints/004_re").resolve()

def rgb_post_q2_ld0_lr1e3_cosine_bdt4_004_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-33-54-304726/checkpoints/004_re").resolve()

def rgb_post_q4_ld0_lr1e3_cosine_bdt4_004_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-33-58-218005/checkpoints/004_re").resolve()

def rgb_post_q6_ld0_lr1e3_cosine_bdt4_004_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-34-01-944119/checkpoints/004_re").resolve()

def rgb_post_q8_ld0_lr1e3_cosine_bdt4_004_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-34-06-491856/checkpoints/004_re").resolve()

def rgb_post_q2_ld0_lr1e3_cosine_bdt4_004_faster_no_mse(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-36-39-672258/checkpoints/004_re").resolve()

def rgb_post_q4_ld0_lr1e3_cosine_bdt4_004_faster_no_mse(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-38-51-447685/checkpoints/004_re").resolve()

def rgb_post_q6_ld0_lr1e3_cosine_bdt4_004_faster_no_mse(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-36-48-284553/checkpoints/004_re").resolve()

def rgb_post_q8_ld0_lr1e3_cosine_bdt4_004_faster_no_mse(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_11-36-52-622125/checkpoints/004_re").resolve()

def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_15-57-26-788776/checkpoints/004_re").resolve()

def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_009(qp):
    return Path("../../filter/logs/train/runs/2025-01-02_15-57-26-788776/checkpoints/009_re").resolve()

def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-03_04-01-36-750425/checkpoints/004_re").resolve()

def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster_no_mse(qp):
    return Path("../../filter/logs/train/runs/2025-01-03_04-01-40-564825/checkpoints/004_re").resolve()

def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_009_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-03_04-01-36-750425/checkpoints/009_re").resolve()

def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_009_faster_no_mse(qp):
    return Path("../../filter/logs/train/runs/2025-01-03_04-01-40-564825/checkpoints/009_re").resolve()

def rgb_pre_from_joint_q2_ld25_lr1e3_cosine_pre_mse_004_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster(qp): ###### 25.01 기고, lic trained
    return Path("../../filter/logs/train/runs/rgb_pre_from_joint_q2_ld25_lr1e3_cosine_pre_mse_004_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster/checkpoints").resolve()

def yuv_post_q_multi_ld0_lr1e3_cosine_bdt4_004(qp):
    return Path("../../filter/logs/train/runs/2025-01-06_01-47-01-632735/checkpoints/004_re").resolve()

def rgb_pre_from_joint_q2_ld25_lr1e3_cosine_pre_mse_004_scratch_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster(qp):
    return Path("../../filter/logs/train/runs/2025-01-08_05-14-57-366424/checkpoints/004_re").resolve()

def rgb_post_q_multi_faster_pcw_finetune(qp): ###### 25.01 기고, post finetuned
    return Path("../../filter/logs/train/runs/rgb_post_q_multi_faster_pcw_finetune/checkpoints/").resolve()
    
def rgb_post_q_multi_ld0_lr1e3_cosine_bdt4_004_yolov3(qp):
    return Path("../../filter/logs/train/runs/2025-01-09_04-35-51-554604/checkpoints/004_re").resolve()

def rgb_pre_from_joint_q2_ld25_lr1e3_cosine_pre_mse_004_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster_pcw(qp):
    return Path("../../filter/logs/train/runs/rgb_pre_from_joint_q2_ld25_lr1e3_cosine_pre_mse_004_post_q_multi_ld0_lr1e3_cosine_bdt4_004_faster_pcw/checkpoints").resolve()

def rgb_pre_rgb_post(qp):
    return Path("../vcmrs/JointFilter/checkpoints/rgb_pre_rgb_post").resolve()

def yuv_pre_rgb_post(qp):
    return Path("../vcmrs/JointFilter/checkpoints/yuv_pre_rgb_post").resolve()