#include "mazeRouter.h"

const int turnCost = 50;

struct Node {
    int d, x, y, dist;
    Node(int d, int x, int y, int dist) : d(d), x(x), y(y), dist(dist) {}
    bool operator < (const Node &rhs) const {
        return dist > rhs.dist;
    }
};

pair<int, int> MazeRouter::Route(const vector<vector<int>> &cost, const int N, const vector<pair<int, int>> &pins, vector<pair<int, int>> &res) {
    auto clocks = clock();
    const int n = pins.size();
    static vector<vector<vector<int>>> vis(2, vector<vector<int>>(N, vector<int> (N, 0))), 
                                      dist(2, vector<vector<int>>(N, vector<int> (N, 0))), prev(2, vector<vector<int>>(N, vector<int> (N, 0)));
    static const int dx[] = {-1, 1, 0, 0};
    static const int dy[] = {0, 0, -1, 1};
    unordered_map<int, int> isPin;
    for(int i = 1; i < n; i++)
        isPin[pins[i].first * N + pins[i].second] = 1;
    res.clear();
    auto computeClocks = clock();
    res.emplace_back(pins[0]);
    int NN = N * N;
    for(int i = 1; i < pins.size(); i++) {
        priority_queue<Node> que;
        for(auto e : res)
            for(int d = 0; d < 2; d++)
                vis[d][e.first][e.second] = i, dist[d][e.first][e.second] = 0, prev[d][e.first][e.second] = -1, que.push(Node(d, e.first, e.second, 0));
        Node found(-1, -1, -1, 0);
        while(!que.empty()) {
            auto o = que.top();
            que.pop();
            int &curx = o.x, &cury = o.y;
            if(isPin[o.x * N + o.y]) {
                found = o;
                isPin[curx * N + cury] = 0;
                break;
            }
            if(o.dist != dist[o.d][curx][cury]) continue;
            for(int d = 0; d < 4; d++) {
                int nx = curx + dx[d], ny = cury + dy[d], nd = d / 2;
                if(!(0 <= nx && nx < N && 0 <= ny && ny < N && cost[nx][ny] < INF)) continue;
                if(vis[nd][nx][ny]  < i || dist[nd][nx][ny] > dist[o.d][curx][cury] + cost[nx][ny] + abs(nd - o.d) * turnCost) {
                    dist[nd][nx][ny] = dist[o.d][curx][cury] + cost[nx][ny] + abs(nd - o.d) * turnCost;
                    prev[nd][nx][ny] = d + 4 * o.d;
                    vis[nd][nx][ny] = i;
                    que.push(Node(nd, nx, ny, dist[nd][nx][ny]));
                }    
            }
        }
        auto o = found;
        if(o.d == -1) return make_pair(-1, -1);
        while(prev[o.d][o.x][o.y] != -1) {
            res.emplace_back(o.x, o.y);
            int d = prev[o.d][o.x][o.y];
            o.d = d >> 2;
            o.x -= dx[d & 3];
            o.y -= dy[d & 3];
        }
    }
    computeClocks = clock() - computeClocks;
    clocks = clock() - clocks;
    return make_pair(clocks, computeClocks);
}