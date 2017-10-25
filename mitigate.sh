#Installing the vim editor
apt-get -y install vim

#Installing the selenium headless browser
#pip install selenium

pip install ipyleaflet
jupyter nbextension enable --py --sys-prefix ipyleaflet

# needed to be able to play videos in the notebook
pip install imageio
pip install moviepy
pip install requests
