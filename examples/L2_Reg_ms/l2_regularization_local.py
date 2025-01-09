import os
import time
import shutil
import platform


t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "l2_regularization/method_test"
fogm_method = (["VSM"], ["VFM"], ["MESM"], ["PGDM"])


base_folder = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_folder, args, t0)

print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

script_extension = ".bat" if platform.system() == "Windows" else ".sh"
script_file = os.path.join(folder, "set" + script_extension)


ganfolder = os.path.join(folder, "l2_regularization.py")
shutil.copyfile(os.path.join(base_folder, "l2_regularization.py"), ganfolder)

with open(script_file, "w") as f:
    k = 0
    for hyper_method in fogm_method:
        k += 1
        print("Comb.{}:".format(k))
        print("hyper_method:", hyper_method)
        f.write("python l2_regularization.py --fo_gm {} \n".format(hyper_method[0]))

if platform.system() != "Windows":
    os.chmod(script_file, 0o775)

if platform.system() == "Windows":
    os.system(script_file)
else:
    os.system(f"bash {script_file}")
