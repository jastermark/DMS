#!/bin/bash


# Image pairs from ScanNet. Evaluation protocol from SuperGlue paper (Sarlin et al.)
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_dkm.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_roma.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500-images.zip

mkdir megadepth1500-images
unzip megadepth1500-images.zip -d megadepth1500-images