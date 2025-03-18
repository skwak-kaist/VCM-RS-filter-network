#!/bin/bash

# 84 * 3  = 252

num_scripts=14
iter=3

echo ${iter}

for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	#echo "#!/bin/bash" > "${filename}" # 신규로 생성할 경우
#	echo "cd .." >> "${filename}" # 신규로 생성할 경우


	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_tvd_tracking_RA_e2e.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done

for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_tvd_tracking_LD_e2e.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done


for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_tvd_tracking_AI_e2e.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done


for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
#	echo "#!/bin/bash" > "${filename}" # 신규로 생성할 경우


	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_tvd_tracking_RA_inner.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done

for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_tvd_tracking_LD_inner.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done


for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_tvd_tracking_AI_inner.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done

