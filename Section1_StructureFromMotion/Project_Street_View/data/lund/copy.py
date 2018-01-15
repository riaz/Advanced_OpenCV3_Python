import os

fl = os.getcwd()

for f in os.listdir(fl):
    if os.path.isfile(f):
        os.system("docker cp " + os.path.realpath(f) +  " 8f60d1b70a4c:/source/OpenSfM/data/lund/" )
    else:
        for fi  in os.listdir(f):
            os.system("docker cp " + "images/" + fi  +  " 8f60d1b70a4c:/source/OpenSfM/data/lund/images" )
