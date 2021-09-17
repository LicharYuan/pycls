import os
"""
when build bench, 
repo will produces many checkpoints, 
you can use this script to keep final checkpionts
"""

def rm_redundant_cp(root_path):
    for root, dirs, files in os.walk(root_path):
        if "checkpoints" in root:
            for ele in files:
                # print(root)
                if ele.split(".")[-1] == "pyth" :
                    cp_filename = ele.split(".")[0]
                    print(cp_filename)
                    cp_epoch = eval(cp_filename[-1])
                    if cp_epoch !=0:
                        # 0 means 10
                        print(cp_epoch)
                        abs_path = os.path.join(root, ele)
                        print("rm:", abs_path)
                        os.remove(abs_path)

            # exit()

if __name__ == "__main__":
    rm_redundant_cp("/mnt/truenas/scratch/liuchun.yuan/projects/pycls/ts/AnyNet/Calibrate")