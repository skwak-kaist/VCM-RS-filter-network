#!/bin/bash

output_path=$1

datasets="SFU TVD_tracking"
encoding_modes="AI_e2e LD_e2e RA_e2e AI_inner LD_inner RA_inner"

seq_SFU="Traffic_2560x1600_30 ParkScene_1920x1080_24 BasketballDrive_1920x1080_50 BQTerrace_1920x1080_60 BasketballDrill_832x480_50 BQMall_832x480_60 PartyScene_832x480_50 \
RaceHorses_832x480_30 BasketballPass_416x240_50 BQSquare_416x240_60 BlowingBubbles_416x240_50 RaceHorses_416x240_30 Kimono_1920x1080_24 Cactus_1920x1080_50"

#seqs="Traffic ParkScene BasketballDrive BQTerrace BasketballDrill BQMall PartyScene RaceHorses BasketballPass BQSquare BlowingBubbles RaceHorses Kimono Cactus"

seq_TVD_tracking="TVD-01_1 TVD-01_2 TVD-01_3 TVD-02_1 TVD-03_1 TVD-03_2 TVD-03_3"

qps="0 1 2 3 4 5"

check_file=$output_path/encoding_check.txt
# 파일 생성
echo "ENCODING_FILE_CHECKING" > $check_file

# Encoding_File_Checking 이라는 문자열을 터미널에 초록색 bold 글씨로 출력
echo -e "\033[1;32m[ ENCODING FILE CHECKER ]\033[0m"

# SFU DATASET 이라는 문자열을 터미널에 흰색 bold로 출력
echo "SFU"
# SFU
for encoding_mode in $encoding_modes; do
    # encoding_mode를 흰색 글씨로 터미널에 출력
    echo ""
    echo -e "\033[37m$encoding_mode\t\033[0m"
    # idx는 1부터 84까지
    for idx in $(seq 0 83); do
        # idx를 14로 나눈 나머지
        seq_idx=$((idx % 14))
        
        # idx를 14로 나눈 몫
        qp=$((idx / 14))
        
        # seq_name은 seq_SFU의 seq_idx번째 원소
        seq_name=$(echo $seq_SFU | cut -d ' ' -f $((seq_idx+1)))
        
        # $output_path/SFU_$encoding_mode/recon에서 seq_name_$qp.yuv 파일이 존재하는지 확인
        # 파일이 존재하면 초록색 글씨로 idx를 출력하고 존재하지 않으면 빨간색 글씨로 idx를 출력
        # 줄바꿈은 없이 출력
        if [ -f $output_path/SFU_$encoding_mode/recon/$seq_name"_qp"$qp.yuv ]; then
            echo -e "\033[1;32m$((idx+1))\033[0m\c"
        else
            echo -e "\033[1;31m$((idx+1))\033[0m\c"
            # 존재하지 않는 파일 명을 check_file에 저장
            echo "SFU_$encoding_mode/recon/$seq_name"_qp"$qp.yuv" >> $check_file
        fi
        # 띄어쓰기 삽입
        echo -e " \c"
    

    done

done

echo -e "\n"

# TVD_tracking 이라는 문자열을 터미널에 흰색 bold로 출력
echo "TVD_tracking"

# TVD_tracking
for encoding_mode in $encoding_modes; do
    # encoding_mode를 흰색 글씨로 터미널에 출력
    echo ""
    echo -e "\033[37m$encoding_mode\t\033[0m"
    # idx는 1부터 42까지
    for idx in $(seq 0 41); do
        # idx를 7로 나눈 나머지
        seq_idx=$((idx % 7))
        
        # idx를 7로 나눈 몫
        qp=$((idx / 7))
        
        # seq_name은 seq_SFU의 seq_idx번째 원소
        seq_name=$(echo $seq_TVD_tracking | cut -d ' ' -f $((seq_idx+1)))
        
        # $output_path/SFU_$encoding_mode/recon에서 seq_name_$qp.yuv 파일이 존재하는지 확인
        # 파일이 존재하면 초록색 글씨로 idx를 출력하고 존재하지 않으면 빨간색 글씨로 idx를 출력
        # 줄바꿈은 없이 출력
        if [ -f $output_path/TVD_tracking_$encoding_mode/recon/qp$qp/$seq_name.yuv ]; then
            echo -e "\033[1;32m$((idx+1))\033[0m\c"
        else
            echo -e "\033[1;31m$((idx+1))\033[0m\c"
            # 존재하지 않는 파일 명을 check_file에 저장
            echo "TVD_tracking_$encoding_mode/recon/qp$qp/$seq_name.yuv" >> $check_file
        fi
        # 띄어쓰기 삽입
        echo -e " \c"
    

    done
done


echo -e "\n"