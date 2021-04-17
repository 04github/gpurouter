#include "mazeRouter.h"
#include "gpuRouter.h"

int N, NumBlks, NumPins;
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

void dfs(int x, int y, vector<vector<int>> &cost) {
    if(cost[x][y] != -1) return;
    cost[x][y] = 0;
    for(int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y+ dy[d];
        if(0 <= nx && nx < N && 0 <= ny && ny < N)
            dfs(nx, ny, cost);
    }
}

int evaluate(const vector<pair<int, int>> &res, vector<vector<int>> cost, const int N) {   
    map<int, int> cnt;
    const int turnCost = 50; 
    int tot = 0;
    for(auto e : res) {
        assert(cost[e.first][e.second] != -1);
        cnt[cost[e.first][e.second]]++;
        tot += cost[e.first][e.second];
        cost[e.first][e.second] = -1;
    }
    int turnCount = 0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) if(cost[i][j] == -1) {
            int test[2] = {0, 0};
            for(int d = 0; d < 4; d++) {
                int x = i + dx[d], y = j + dy[d];
                test[d / 2] |= (0 <= x && x < N && 0 <= y && y < N && cost[x][y] == -1);
            }
            turnCount += test[0] && test[1];
        }
    dfs(res[0].first, res[0].second, cost);
    tot += turnCount * turnCost;
    for(auto e : res) 
        if(cost[e.first][e.second] == -1) tot = -1;
    cerr << "Path analysis------- " << endl;
    cerr << "cost count sum percent(number) percent(length)" << endl;
    for(auto e : cnt)
        cerr << e.first << ' ' << e.second << ' ' << e.first * e.second << ' ' << e.second * 100.0 / res.size() << "% " << e.first * e.second * 100.0 / tot << "%" << endl;
    cerr << "turn " << turnCount << ' ' << turnCost * turnCount << " N/A " << turnCost * turnCount * 100.0 / tot << "%" << endl;
    cerr << "total N/A " << tot << endl;
    cerr << "Path analysis^^^^^^^ " << endl;
    return tot;
}

void output(vector<pair<int, int>> &res, const vector<vector<int>> &cost, const int N) {
    sort(res.begin(), res.end());
    for(auto e : res)
        cerr << e.first << ' ' << e.second << "   " << cost[e.first][e.second] << endl;
}

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
    cin >> NumPins;
    vector<pair<int, int>> pins(NumPins), gpures, cpures;
    for(int i = 0; i < NumPins; i++)
        cin >> pins[i].first >> pins[i].second;
    MazeRouter mazeRouter;
    auto gputime = Route(cost, N, pins, gpures);
    auto cputime = mazeRouter.Route(cost, N, pins, cpures);
    cerr << evaluate(gpures, cost, N) << ' ' << evaluate(cpures, cost, N) << ' ' << 
            gputime.second << ' ' << cputime.second << ' ' << gputime.first << ' ' << cputime.first << endl;
    
    cout << N << endl;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++)
            cout << (cost[i][j] == INF ? -1 : cost[i][j]) << ' ';
        cout << endl;
    }
    cout << NumPins << endl;
    for(auto e : pins)
        cout << e.first << ' ' << e.second << endl;
    cout << gpures.size() << endl;
    for(auto e : gpures)
        cout << e.first << ' ' << e.second << endl;
    cout << cpures.size() << endl;
    for(auto e : cpures)
        cout << e.first << ' ' << e.second << endl;
  
    return 0;
}