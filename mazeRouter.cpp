#include "mazeRouter.h"

pair<int, int> MazeRouter::Route(const vector<vector<int>> &cost, const int N, const vector<pair<int, int>> &pins, vector<pair<int, int>> &res) {
    auto clocks = clock();
    const int n = pins.size();
    static vector<vector<int>> vis(N, vector<int> (N, 0)), dist(N, vector<int> (N)), prev(N, vector<int> (N));
    static const int dx[] = {-1, 0, 0, 1};
    static const int dy[] = {0, -1, 1, 0};
    set<int> pinSet;
    for(int i = 1; i < n; i++)
        pinSet.insert(pins[i].first * N + pins[i].second);
    res.clear();
    auto computeClocks = clock();
    res.emplace_back(pins[0]);
    for(int i = 1; i < pins.size(); i++) {
        priority_queue<pair<int, int>> que;
        for(auto e : res)
            vis[e.first][e.second] = i, dist[e.first][e.second] = 0, prev[e.first][e.second] = -1, que.push(make_pair(0, e.first * N + e.second));
        int x = -1, y = -1;
        while(!que.empty()) {
            auto o = que.top();
            que.pop();
            int curx = o.second / N, cury = o.second % N;
            //cerr << curx << ' ' << cury << endl;
            if(pinSet.find(o.second) != pinSet.end()) {
                x = curx;
                y = cury;
                pinSet.erase(o.second);
                break;
            }
            if(o.first != -dist[curx][cury]) continue;
            for(int d = 0; d < 4; d++) {
                int nx = curx + dx[d], ny = cury + dy[d];
                if(0 <= nx && nx < N && 0 <= ny && ny < N && cost[nx][ny] < INF && (vis[nx][ny]  < i || dist[nx][ny] > dist[curx][cury] + cost[nx][ny])) {
                    dist[nx][ny] = dist[curx][cury] + cost[nx][ny];
                    prev[nx][ny] = d;
                    vis[nx][ny] = i;
                    que.push(make_pair(-dist[nx][ny], nx * N + ny));
                }    
            }
        }
        if(x == -1) return make_pair(-1, -1);
        while(prev[x][y] != -1) {
            res.emplace_back(x, y);
            int d = prev[x][y];
            x -= dx[d];
            y -= dy[d];
        }
    }
    computeClocks = clock() - computeClocks;
    clocks = clock() - clocks;
    return make_pair(clocks, computeClocks);
}