#!/bin/sh

ffmpeg -r 50 -pattern_type glob -i '*.png' -c:v libx264 -vf "format=yuv420p" out.mp4
