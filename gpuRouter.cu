#include "gpuRouter.h"

__managed__ int success;
__managed__ int extraTurns;
__managed__ int N;
const int maxTurns = 200, ExtraTurns = 10;
const int ThreadsPerBlock = 1 << 10;
__device__ const int dx[] = {0, 0, -1, 1};
__device__ const int dy[] = {-1, 1, 0, 0};
__device__ const int turnCost = 50;
int totTurns, cntTurns;

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
__global__ void Min(int *x, int *y) {
/*    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = y[idx] + turnCost;
    if(val < x[idx])
        x[idx] = val;*/
    x[blockIdx.x * blockDim.x + threadIdx.x] = min(x[blockIdx.x * blockDim.x + threadIdx.x], y[blockIdx.x * blockDim.x + threadIdx.x] + turnCost); 
}

__global__ void Check(int *res, int *map0, int *map1, int *cost, int *pins, int n) {
    int minDist = INF, x, y, dir;
    for(int i = 0; i < n; i++)
        if(res[pins[i]]) minDist = min(minDist, min(map0[pins[i]], map1[pins[i]]));
    if(minDist == INF || extraTurns--) return;
    success = 1;
    for(int i = 0; i < n; i++) {
        if(res[pins[i]] && map0[pins[i]] == minDist) {
            x = pins[i] / N;
            y = pins[i] % N;
            dir = 0;        
            break;    
        }        
        if(res[pins[i]] && map1[pins[i]] == minDist) {
            x = pins[i] / N;
            y = pins[i] % N;
            dir = 1;
            break;
        }
    }
    if(dir) {
        int *temp = map0;
        map0 = map1;
        map1 = temp;
    }
    while(res[x * N + y] != 0) {    
        int idx = x * N + y;
        if(map1[idx] + turnCost <= map0[idx]) {
            int *temp = map0;
            map0 = map1;
            map1 = temp;
            dir ^= 1;
        }
        res[idx] = 0;
        for(int d = 0; d < 2; d++) {
            int nx = x + dx[d + dir * 2], ny = y + dy[d + dir * 2];
            if(0 <= nx && nx < N && 0 <= ny && ny < N && map0[nx * N + ny] + cost[idx] <= map0[idx]) {
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
    int *cudaCost, *cudaMap0, *cudaMap1, *cudaRes, *cudaPins, *copyTemp = new int[N * N];
    cudaMalloc(&cudaCost, N * N * sizeof(int));
    cudaMalloc(&cudaMap0, N * N * sizeof(int));
    cudaMalloc(&cudaMap1, N * N * sizeof(int));
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
    totTurns = cntTurns = 0;
    Init<<<N, N>>> (cudaRes, pins[0].first * N + pins[0].second);
    //int cnta = 0, cntb = 0, cntc = 0;
    for(int i = 1; i < n; i++) {
        //auto cntbTimer = clock();
        Copy<<<N, N>>> (cudaMap0, cudaRes);
        Copy<<<N, N>>> (cudaMap1, cudaRes);
        //cudaDeviceSynchronize();
        //cntb += clock() - cntbTimer;
        success = 0;
        extraTurns = ExtraTurns;
        for(int turns = 0; turns < maxTurns; turns++) {
            //auto cntaTimer = clock();
            ShortestPathLR<<<N, N>>> (cudaMap0, cudaCost);
            //cudaDeviceSynchronize();
            //cnta += clock() - cntaTimer;
            Min<<<N, N>>> (cudaMap1, cudaMap0);
            //cudaDeviceSynchronize();
            //cntc += clock() - cntaTimer;
            ShortestPathUD<<<N, N>>> (cudaMap1, cudaCost);
            Min<<<N, N>>> (cudaMap0, cudaMap1);
            Check<<<1, 1>>> (cudaRes, cudaMap0, cudaMap1, cudaCost, cudaPins, n);
            cudaDeviceSynchronize();
            totTurns++;
            if(success) break;
        }
        cntTurns++;
        cerr << success << " success " << endl;
        if(!success) break;
    }
    //cerr << "comparison " << cnta << ' ' << cntb << ' ' << cntc - cnta << endl;
    computeClocks = clock() - computeClocks;
    
    cerr << "average turns: " << 1.0 * totTurns / cntTurns << endl;
    
    cudaMemcpy(copyTemp, cudaRes, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    res.clear();
    for(int i = 0; i < N; i++) 
        for(int j = 0; j < N; j++)
            if(copyTemp[i * N + j] == 0)
                res.emplace_back(i, j);
    cudaFree(cudaCost);
    cudaFree(cudaMap0);
    cudaFree(cudaMap1);
    cudaFree(cudaRes);
    cudaFree(cudaPins);
    delete[] copyTemp;
    clocks = clock() - clocks;
    return success ? make_pair((int) clocks, (int) computeClocks) : make_pair(-1, -1);
}
