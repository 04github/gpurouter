#include "gpuRouter.h"

__managed__ int success;
__managed__ int extraTurns;
__managed__ int N;
const int maxTurns = 100, ExtraTurns = 0;
const int ThreadsPerBlock = 1 << 10;
__device__ const int dx[] = {-1, 0, 0, 1};
__device__ const int dy[] = {0, -1, 1, 0};

__global__ void ShortestPathLR(int *s, int *c) {
    __shared__ int Lc[ThreadsPerBlock], Rc[ThreadsPerBlock], LRs[ThreadsPerBlock];
    LRs[threadIdx.x] = s[blockIdx.x * blockDim.x + threadIdx.x];
    Lc[threadIdx.x] = Rc[threadIdx.x] = c[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    for(int d = 0; (1 << d) < blockDim.x; d++) {
        if(threadIdx.x >> d & 1) {
            int last = (threadIdx.x >> d << d) - 1;
            LRs[threadIdx.x] = min(LRs[threadIdx.x], LRs[last] + Lc[threadIdx.x]);
            Lc[threadIdx.x] = min(Lc[threadIdx.x] + Lc[last], INF);
            LRs[blockDim.x - 1 - threadIdx.x] = min(LRs[blockDim.x - 1 - threadIdx.x], LRs[blockDim.x - 1 - last] + Rc[blockDim.x - 1 - threadIdx.x]);
            Rc[blockDim.x - 1 - threadIdx.x] = min(Rc[blockDim.x - 1 - threadIdx.x] + Rc[blockDim.x - 1 - last], INF);
        }
        __syncthreads();
    }
    s[blockIdx.x * blockDim.x + threadIdx.x] = LRs[threadIdx.x];
}
__global__ void ShortestPathUD(int *s, int *c) {
    __shared__ int Uc[ThreadsPerBlock], Dc[ThreadsPerBlock], UDs[ThreadsPerBlock];
    UDs[threadIdx.x] = s[threadIdx.x * blockDim.x + blockIdx.x];
    Uc[threadIdx.x] = Dc[threadIdx.x] = c[threadIdx.x * blockDim.x + blockIdx.x];
    __syncthreads();
    for(int d = 0; (1 << d) < blockDim.x; d++) {
        if(threadIdx.x >> d & 1) {
            int last = (threadIdx.x >> d << d) - 1;
            UDs[threadIdx.x] = min(UDs[threadIdx.x], UDs[last] + Uc[threadIdx.x]);
            Uc[threadIdx.x] = min(Uc[threadIdx.x] + Uc[last], INF);
            UDs[blockDim.x - 1 - threadIdx.x] = min(UDs[blockDim.x - 1 - threadIdx.x], UDs[blockDim.x - 1 - last] + Dc[blockDim.x - 1 - threadIdx.x]);
            Dc[blockDim.x - 1 - threadIdx.x] = min(Dc[blockDim.x - 1 - threadIdx.x] + Dc[blockDim.x - 1 - last], INF);
        }
        __syncthreads();
    }
    s[threadIdx.x * blockDim.x + blockIdx.x] = UDs[threadIdx.x];
}

__global__ void Init(int *x, int pos) { x[blockIdx.x * blockDim.x + threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x == pos ? 0 : INF; }
__global__ void Copy(int *x, int *y) { x[blockIdx.x * blockDim.x + threadIdx.x] = y[blockIdx.x * blockDim.x + threadIdx.x]; }

__global__ void Check(int *res, int *map, int *cost, int *pins, int n) {
    int minDist = INF, x, y;
    for(int i = 0; i < n; i++)
        if(res[pins[i]]) minDist = min(minDist, map[pins[i]]);
    if(minDist == INF || extraTurns--) return;
    success = 1;
    for(int i = 0; i < n; i++)
        if(res[pins[i]] && map[pins[i]] == minDist) {
            x = pins[i] / N;
            y = pins[i] % N;
        }
    while(res[x * N + y] != 0) {    
        int idx = x * N + y;
        res[idx] = 0;
        for(int d = 0; d < 4; d++) {
            int nx = x + dx[d], ny = y + dy[d];
            if(0 <= nx && nx < N && 0 <= ny && ny < N && map[nx * N + ny] + cost[idx] <= map[idx]) {
                x = nx;
                y = ny;
                break;
            }
        }
    }
}

pair<int, int> Route(const vector<vector<int>> &cost, const int N, const vector<pair<int, int>> &pins, vector<pair<int, int>> &res) {
    ::N = N;
    auto clocks = clock();
    const int n = pins.size();
    int *cudaCost, *cudaMap, *cudaRes, *cudaPins, *copyTemp = new int[N * N];
    cudaMalloc(&cudaCost, N * N * sizeof(int));
    cudaMalloc(&cudaMap, N * N * sizeof(int));
    cudaMalloc(&cudaRes, N * N * sizeof(int));
    cudaMalloc(&cudaPins, n * sizeof(int));
    for(int i = 0; i < N; i++) 
        for(int j = 0; j < N; j++)
            copyTemp[i * N + j] = cost[i][j];
    cudaMemcpy(cudaCost, copyTemp, N * N * sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < n; i++)
        copyTemp[i] = pins[i].first * N + pins[i].second;
    cudaMemcpy(cudaPins, copyTemp, n * sizeof(int), cudaMemcpyHostToDevice);
    
    auto computeClocks = clock();
    Init<<<N, N>>> (cudaRes, pins[0].first * N + pins[0].second);
    cudaDeviceSynchronize();
    for(int i = 1; i < n; i++) {
        Copy<<<N, N>>> (cudaMap, cudaRes);
        success = 0;
        extraTurns = ExtraTurns;
        for(int turns = 0; turns < maxTurns; turns++) {
            ShortestPathLR<<<N, N>>> (cudaMap, cudaCost);
            ShortestPathUD<<<N, N>>> (cudaMap, cudaCost);
            Check<<<1, 1>>> (cudaRes, cudaMap, cudaCost, cudaPins, n);
            cudaDeviceSynchronize();
            if(success) break;
        }
        if(!success) break;
    }
    
    computeClocks = clock() - computeClocks;
    
    cudaMemcpy(copyTemp, cudaRes, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    res.clear();
    for(int i = 0; i < N; i++) 
        for(int j = 0; j < N; j++)
            if(copyTemp[i * N + j] == 0)
                res.emplace_back(i, j);
    cudaFree(cudaCost);
    cudaFree(cudaMap);
    cudaFree(cudaRes);
    cudaFree(cudaPins);
    delete[] copyTemp;
    clocks = clock() - clocks;
    return success ? make_pair((int) clocks, (int) computeClocks) : make_pair(-1, -1);
}
