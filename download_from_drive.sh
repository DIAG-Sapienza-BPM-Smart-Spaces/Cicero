#!/bin/bash
fileid="1EYQnrFpY2vGQFSHaHRcILlxE2drwX8ci"
filename="ANON_merge23_04.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
