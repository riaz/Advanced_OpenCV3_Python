import os
import sys

c_id = "8f60d1b70a4c"

print(str(sys.argv))

if len(sys.argv) > 2:
    c_id = sys.argv[1]
    img_folder = sys.argv[2]

print(c_id)
print(img_folder)

fl = os.getcwd()

for f in os.listdir(fl):
    if os.path.isfile(f):
        os.system("docker cp " + os.path.realpath(f) +  " {0}:/source/OpenSfM/data/{1}/".format(c_id,img_folder) )
    else:
        for fi  in os.listdir(f):
            os.system("docker cp " + "images/" + fi  +  " {0}:/source/OpenSfM/data/{1}/images/{2}".format(c_id,img_folder,fi) )
