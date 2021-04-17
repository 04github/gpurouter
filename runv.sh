mkdir -p output
nvcc main.cpp mazeRouter.cpp gpuRouter.cu -o executable -std=c++11 -O2 
./executable < benchmarks/$1.txt > output/$1output.txt
python3 visualizer.py -i $1