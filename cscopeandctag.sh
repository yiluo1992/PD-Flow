#!/bin/sh

find ~/SceneFlow/PD-Flow -name "*.h" -o -name "*.c" -o -name "*.cpp" -o -name "*.cu" > cscope.files
find ~/Software/opencv-3.0.0/modules -name "*.h" -o -name "*.c" -o -name "*.cpp" >> cscope.files

cscope -bkq -i cscope.files
