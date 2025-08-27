struct Point {
    long long x, y;
};

__int128 cross(const Point& A, const Point& B, const Point& C) {
    return (__int128)(B.x - A.x) * (C.y - A.y) - (__int128)(B.y - A.y) * (C.x - A.x);
}

bool pointInConvexPolygon(const vector<Point>& poly, const Point& P) {
    int n = poly.size();
    int sign = 0;
    for (int i = 0; i < n; i++) {
        __int128 c = cross(poly[i], poly[(i+1)%n], P);
        if (c == 0) continue;              // 在边上也算 inside
        if (sign == 0) sign = (c > 0 ? 1 : -1);
        else if ((c > 0) != (sign > 0)) return false;
    }
    return true;
}