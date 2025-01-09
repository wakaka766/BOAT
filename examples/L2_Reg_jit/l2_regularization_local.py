import os
import time
import shutil
import platform


t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = "l2_regularization/method_test"
dynamic_methodlist = (
    ["NGD"],
    ["DI", "NGD"],
    ["GDA", "NGD"],
    ["GDA", "NGD", "DI"],
    ["DI", "NGD", "GDA"],
)
dynamic_method_dm = (["DM"], ["DM", "GDA"])
hyper_methodlist = (
    ["CG"],
    ["CG", "PTT"],
    ["RAD"],
    ["RAD", "PTT"],
    ["RAD", "RGT"],
    ["PTT", "RAD", "RGT"],
    ["FD"],
    ["FD", "PTT"],
    ["NS"],
    ["NS", "PTT"],
    ["IGA"],
    ["IGA", "PTT"],
)
hyper_method_dm = (["RAD"], ["CG"])
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
    for dynamic_method in dynamic_methodlist:
        for hyper_method in hyper_methodlist:
            k += 1
            print("Comb.{}:".format(k))
            print("dynamic_method:", dynamic_method, " hyper_method:", hyper_method)
            f.write(
                "python l2_regularization.py --dynamic_method {} --hyper_method {} \n".format(
                    ",".join([dynamic for dynamic in dynamic_method]),
                    ",".join([hyper for hyper in hyper_method]),
                )
            )

    for dynamic_method in dynamic_method_dm:
        for hyper_method in hyper_method_dm:
            k += 1
            print("Comb.{}:".format(k))
            print("dynamic_method:", dynamic_method, " hyper_method:", hyper_method)
            f.write(
                "python l2_regularization.py --dynamic_method {} --hyper_method {} \n".format(
                    ",".join([dynamic for dynamic in dynamic_method]),
                    ",".join([hyper for hyper in hyper_method]),
                )
            )

    for hyper_method in fogm_method:
        k += 1
        print("Comb.{}:".format(k))
        print("hyper_method:", hyper_method)
        f.write("python l2_regularization.py --fo_gm {} \n".format(hyper_method[0]))

if platform.system() != "Windows":
    os.chmod(script_file, 0o775)

import subprocess

with open(script_file, "r") as f:
    commands = f.readlines()

for command in commands:
    print(f"Running: {command.strip()}")
    subprocess.run(command.strip(), shell=True)
