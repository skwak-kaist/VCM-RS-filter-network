# introduction
* VCM-RS with joint filter network
  * current version: v0.12



1. Place 'Data' folder at the top level of the project (same as VCM-RS)

2. Install the conda env. (same as VCM-RS)

3. Place pre-trained model at vcmrs/Jointfilter/checkpoints

   * e.g., vcmrs/Jointfilter/checkpoints/yuv_pre/resnet50/prefilter.pth

4. Place vcm-ctc at the same level of this project 

   * e.g., 

   * Project_VCM

     ㄴ VCM-RS

     ㄴ vcm-ctc

* Pre-filter
  * Run Scripts/run_prefilter.sh
    * adjust the arguments properly
      * component-order: TCSJRBD_SRCPTPB (fixed)
      * gpu-ids, num-worker-per-gpu: adjust according to the PC specifications
      * selection-algorithm: please refer to the sub-folder names of 'checkpoints'
      * pre-domain, post-domain: YUV and RGB (fixed)
* Post-filter
  * Run Sciprts/run_postfilter.sh
  * OR, if you already have the anchor's bitstream, 
    * Run Scripts/VTM_InnerCodec/decode_all_postfilter.sh
    * adjust filter mode and decode_folder
  * 현재 Post Filter는 모든 시퀀스에 다 적용됨. 
  * 해상도에 따라 Post filter의 동작을 제어하기 위해서는 vcmrs/Jointfilter/filter.py 에서
    42번 라인에 주석처리 되어있는 post_flag를 활성화 시키고, 43번 라인의 hard coding을 삭제해야 함















