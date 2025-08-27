#include <cstring>
#include <iostream>

char a[200] = "ADABBC";
char b[200] = "DBPCA";
int f[200][200];
int p[200][200];
int m, n;
void LCS() {
    int i, j;
    m = strlen(a);
    n = strlen(b);
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (a[i-1] == b[j-1]) {
                f[i][j] = f[i-1][j-1] + 1;
                p[i][j] = 1; // 左上方
            } else if (f[i][j-1] > f[i-1][j]) {
                f[i][j] = f[i][j-1];
                p[i][j] = 2; // 左边
            } else {
                f[i][j] = f[i-1][j];
                p[i][j] = 3; // 上边
            }
        }
    }
    cout << f[m][n] << '\n';
}

void getLCS() {
    int i, j, k;
    char s[200];
    i = m;
    j = n;
    k = f[m][n];
    while (i > 0 && j > 0) {
        if (p[i][j] == 1) {
            s[k-1] = a[i-1];
            i--;
            j--;
            k--;
        } else if (p[i][j] == 2) {
            j--;
        } else {
            i--;
        }
    }
    for (i = 0; i < f[m][n]; i++) {
        cout << s[i];
    }
}
int main() {
    LCS();
    getLCS();
    return 0;
}