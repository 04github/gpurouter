#include "mazeRouter.h"
#include "gpuRouter.h"

int N, NumBlks, NumPins;

int main() {
    
    cin >> N;
    vector<vector<int>> cost(N, vector<int> (N));
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            cin >> cost[i][j];
    cin >> NumBlks;
    for(int i = 0; i < NumBlks; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        for(int a = x1; a <= x2; a++)
            for(int b = y1; b <= y2; b++)
                cost[a][b] = INF;
    }
    int gpuwl = 0, cpuwl = 0;
    cin >> NumPins;
    vector<pair<int, int>> pins(NumPins), gpures, cpures;
    for(int i = 0; i < NumPins; i++)
        cin >> pins[i].first >> pins[i].second;
    MazeRouter mazeRouter;
    auto gputime = Route(cost, N, pins, gpures);
    auto cputime = mazeRouter.Route(cost, N, pins, cpures);
    for(auto e : gpures)
        gpuwl += cost[e.first][e.second];
    for(auto e : cpures)
        cpuwl += cost[e.first][e.second];
    cerr << gpuwl << ' ' << cpuwl << ' ' << gputime.second << ' ' << cputime.second << ' ' << gputime.first << ' ' << cputime.first << endl;
    
  
    return 0;
}