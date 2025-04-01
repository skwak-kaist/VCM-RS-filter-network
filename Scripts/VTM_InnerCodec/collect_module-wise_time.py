import os, sys,glob

output_path=sys.argv[1]

folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]

modules = ["JointFilter", "SpatialResample", "ROI", "Colorize", "TemporalResample", "PostFilter", "BitDepthTruncation"]

with open("time_results.txt", "w") as f:
    f.write("")
    
for folder in folders: 
    print(folder)
    
    logfiles = glob.glob(os.path.join(output_path, folder, "decoding_log", "*.log"))
    
    for logfile in logfiles:
        log_name = os.path.basename(logfile)
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Inner decoding done" in line:
                    time = line.split("Time = ")[1].split("(s)")[0]
                    with open("time_results.txt", "a") as f:
                        f.write(f"{folder} {log_name} Inner_decoding: "+ time + "\n")
                
                for module in modules:
                    if f"{module} at decoder done" in line:
                        time = line.split("Time = ")[1].split("(s)")[0]
                        with open("time_results.txt", "a") as f:
                            f.write(f"{folder} {log_name} {module}: "+ time  + "\n")
                
                if "Decoding completed in" in line:
                    time = line.split("Decoding completed in ")[1].split(" seconds")[0]
                    with open("time_results.txt", "a") as f:
                        f.write(f"{folder} {log_name} Decoding_total: "+ time + "\n")
                
            f.close()