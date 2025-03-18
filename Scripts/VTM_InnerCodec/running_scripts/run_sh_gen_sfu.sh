#!/bin/bash

# 84 * 3  = 252

num_scripts=14
iter=6

echo ${iter}

for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	echo "#!/bin/bash" > "${filename}"
	echo "cd .." >> "${filename}"
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_sfu_RA_e2e.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done

for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_sfu_LD_e2e.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done


for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_sfu_AI_e2e.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done


for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
#	echo "#!/bin/bash" > "${filename}"
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_sfu_RA_inner.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done

for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_sfu_LD_inner.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done


for ((i=1; i<=num_scripts; i++)); do
	filename="run${i}.sh"
	initial_arg=${i}
	
	for ((k=1; k<=${iter}; k++)); do
		echo "python encode_sfu_AI_inner.py ${initial_arg}" >> "${filename}"
		
		# 인자 값 증가
		initial_arg=$((initial_arg + $num_scripts))
	done
	
	chmod +x "${filename}"
done



