#include<iostream>
#include<cmath>
#include<ctime>
#include<cstdio>
#include<set>
#include<vector>
#include<cstdlib>
using namespace std;



int main() {

    srand(time(NULL));
    
    const int N = 1 << 10, maxCost = 1000, numBlockages = 5, numPins = 30;
    vector<vector<int>> mp(N, vector<int> (N));
    set<pair<int, int>> used;
    
    cout << N << endl;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) {
            int d = rand() % 100 - 90;
            if(d <= 0)
                mp[i][j] = rand() % 3 + 1;
            else
                mp[i][j] = 3 + (1 << d);
            printf("%d%c", mp[i][j], " \n"[j + 1 == N]);
        }
    cout << numBlockages << endl;
    for(int i = 0; i < numBlockages; i++) {
        int width = rand() % min(100, N / 4) + 1, height = rand() % min(100, N / 4) + 1;
        int x = rand() % (N - width), y = rand() % (N - height);
        cout << x << ' ' << y << ' ' << x + width << ' ' << y + height << endl;
        for(int dx = 0; dx < width; dx++)
            for(int dy = 0; dy < height; dy++)
                used.insert(make_pair(x + dx, y + dy));
    }
    cout << numPins << endl;
    for(int k = 0; k < numPins; k++) {
        int x = rand() % N, y = rand() % N;
        while(used.find(make_pair(x, y)) != used.end())
            x = rand() % N, y = rand() % N;
        used.insert(make_pair(x, y));
        cout << x << ' ' << y << endl;
    }


    return 0;
}