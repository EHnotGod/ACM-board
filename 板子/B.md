# EH整理的の板子（有一半不是自己写的）

这是题目对照。

[TOC]

## A-基础算法

### A1 高精度加法

**原题描述：**

高精度加法，相当于 a+b problem，**不用考虑负数**。

**输入格式：**

分两行输入。*a*,*b*<=$$10^{500}$$。

**输出格式：**

输出只有一行，代表 *a*+*b* 的值。

**输入输出样例：**

输入 #1

```
1
1
```

输出 #1

```
2
```

输入 #2

```
1001
9099
```

输出 #2

```
10100
```

**解析：**时间复杂度O(n)

### A2 高精度减法

**题目描述**

高精度减法。

**输入格式**

两个整数 *a*,*b*（第二个可能比第一个大）。

**输出格式**

结果（是负数要输出负号）。

**输入输出样例**

输入 #1

```
2
1
```

输出 #1

```
1
```

**解析：**时间复杂度O(n)



### A3 高精度乘法

**题目描述**

高精度乘法模板题。

**输入格式**

两个非负整数 *a*,*b*，求它们的乘积。

**输出格式**

输出一个非负整数表示乘积。每个非负整数不超过$$10^{2000}$$

**输入输出样例**

输入 #1

```
2
1
```

输出 #1

```
2
```

**解析：**时间复杂度O($$n^{2}$$)

### A4 高精度除法

**题目描述**

输入两个整数 *a*,*b*，输出它们的商。

**输入格式**

两行，第一行是被除数a，第二行是除数b。

0 ≤ *a* ≤ $$10^{5000}$$，1 ≤*b*≤ $$10^{9}$$

**输出格式**

一行，商的整数部分。

**输入输出样例**

输入 #1

```
2
1
```

输出 #1

```
1
```

**解析：**时间复杂度O($$n$$)

### A5 二分查找

**题目描述**

输入 $n$ 个不超过 $10^9$ 的单调不减的（就是后面的数字不小于前面的数字）非负整数 $a_1,a_2,\dots,a_{n}$，然后进行 $m$ 次询问。对于每次询问，给出一个整数 $q$，要求输出这个数字在序列中第一次出现的编号，如果没有找到的话输出 $-1$ 。

**输入格式**

第一行 $2$ 个整数 $n$ 和 $m$，表示数字个数和询问次数。

第二行 $n$ 个整数，表示这些待查询的数字。

第三行 $m$ 个整数，表示询问这些数字的编号，从 $1$ 开始编号。

**输出格式**

输出一行，$m$ 个整数，以空格隔开，表示答案。

**输入输出样例**

输入 #1

```
11 3
1 3 3 3 5 7 9 11 13 15 15
1 3 6
```

输出 #1

```
1 2 -1
```

数据保证，$1 \leq n \leq 10^6$，$0 \leq a_i,q \leq 10^9$，$1 \leq m \leq 10^5$

本题输入输出量较大，请使用较快的 IO 方式。



**解析：**时间复杂度O($$mlog(n)$$)

对于py，使用bisect_left即可做到快速二分。



### A6 二分答案

**题目描述**

木材厂有 $n$ 根原木，现在想把这些木头切割成 $k$ 段长度**均**为 $l$ 的小段木头（木头有可能有剩余）。请求出 $l$ 的最大值。

确保长度都是正整数。

例如有两根原木长度分别为 $11$ 和 $21$，要求切割成等长的 $6$ 段，很明显能切割出来的小段木头长度最长为 $5$。

**输入格式**

第一行是两个正整数 $n,k$，分别表示原木的数量，需要得到的小段的数量。

接下来 $n$ 行，每行一个正整数 $L_i$，表示一根原木的长度。

**输出格式**

仅一行，即 $l$ 的最大值。

如果连 $\text{1cm}$ 长的小段都切不出来，输出 `0`。

**输入输出样例**

输入 #1

```
3 7
232
124
456
```

输出 #1

```
114
```

对于 $100\%$ 的数据，有 $1\le n\le 10^5$，$1\le k\le 10^8$，$1\le L_i\le 10^8(i\in[1,n])$。



**解析：**时间复杂度O($$nlog(max(L_i))$$)

与传统的二分不同，这题不是n次询问，每项二分，而是直接二分答案，这与校省选的图书馆那题很像。

### A7 分数规划

![image-20250416084038519](image-20250416084038519.png)

### A8 前缀和

**题目描述**

给定 $n$ 个正整数组成的数列 $a_1, a_2, \cdots, a_n$ 和 $m$ 个区间 $[l_i,r_i]$，分别求这 $m$ 个区间的区间和。

对于所有测试数据，$n,m\le10^5,a_i\le 10^4$

**输入格式**

第一行，为一个正整数 $n$ 。

第二行，为 $n$ 个正整数 $a_1,a_2, \cdots ,a_n$

第三行，为一个正整数 $m$ 。

接下来 $m$ 行，每行为两个正整数 $l_i,r_i$ ，满足$1\le l_i\le r_i\le n$

**输出格式**

共 $m$ 行。

第 $i$ 行为第 $i$ 组答案的询问。

**输入输出样例**

输入 #1

```
4
4 3 2 1
2
1 4
2 3
```

输出 #1

```
10
5
```

对于 $100 \%$ 的数据：$1 \le n, m\le 10^5$，$1 \le a_i\le 10^4$

**解析：**时间复杂度$$O(n)$$

把每项的和加起来，然后利用相减的方式得出和的值。适用于静态的区间求和，应用面非常广泛。

#### 二维前缀和

**题目描述**

在一个 $n\times m$ 的只包含 $0$ 和 $1$ 的矩阵里找出一个不包含 $0$ 的最大正方形，输出边长。

**输入格式**

输入文件第一行为两个整数 $n,m(1\leq n,m\leq 100)$，接下来 $n$ 行，每行 $m$ 个数字，用空格隔开，$0$ 或 $1$。

**输出格式**

一个整数，最大正方形的边长。

**输入输出样例**

输入 #1

```
4 4
0 1 1 1
1 1 1 0
0 1 1 0
1 1 0 1
```

输出 #1

```
2
```

**解析：**EH不想写了

![img](10d70759f214e4aa0b8d1d762504e02c.jpg)

![image-20250416084901565](image-20250416084901565.png)



### A10 差分

**题目描述**

给定一个长度为 $n$ 的数列 ${a_1,a_2,\cdots,a_n}$，每次可以选择一个区间$[l,r]$，使这个区间内的数都加 $1$ 或者都减 $1$。 

请问至少需要多少次操作才能使数列中的所有数都一样，并求出在保证最少次数的前提下，最终得到的数列有多少种。

**输入格式**

第一行一个正整数 $n$   
接下来 $n$ 行,每行一个整数,第 $i+1 $行的整数表示 $a_i$。

**输出格式**

第一行输出最少操作次数   
第二行输出最终能得到多少种结果

**输入输出样例**

输入 #1

```
4
1
1
2
2
```

输出 #1

```
1
2
```

对于 $100\%$ 的数据，$n\le 100000, 0 \le a_i \le 2^{31}$。

**解析：**时间复杂度$$O(n)$$

差分与前缀和相对，一个是区间求，一个是区间操作，好在都是静态的，不然就需要线段树了。

### A12 ST表

![image-20250416090732094](image-20250416090732094.png)

![image-20250416090758400](image-20250416090758400.png)

![image-20250416090830823](image-20250416090830823.png)

### A13 快速排序、第k小的数

![image-20250416091013120](image-20250416091013120.png)

### A14 归并排序、逆序对

![image-20250416091200219](image-20250416091200219.png)

### A15 堆

![image-20250416091257702](image-20250416091257702.png)

**解析：**好吧，EH不摆了。

py里的堆可以用heapq高效实现，你不需要知道它是什么，什么是堆（优先队列）。这样吧，你把一个py里的列表q想象成一个盒子，你每次花费logn的时间往里面丢东西（heapq.heappush(q, ai)），它总是会弹出盒子里最小那个数，查询方式为q[0]。你也可以花费logn的时间把这个最小数取出来（heapq.heappop(q)），此时其他数会成为最小的数并置顶。

heapq默认小根堆（弹出最小数），你可以通过加个负号让它变成大根堆。

### A16 对顶堆

![image-20250416092013488](image-20250416092013488.png)

**解析：**

![image-20250416092041815](image-20250416092041815.png)

### A17 距离之和最小、中位数

![image-20250416092236944](image-20250416092236944.png)

### A18 双指针（定量）

![image-20250416092315553](image-20250416092315553.png)

### A19 双指针（定性）

![image-20250416092348763](image-20250416092348763.png)

![image-20250416092410751](image-20250416092410751.png)

### A20 双指针（异或释放）

![image-20250416092446455](image-20250416092446455.png)

![image-20250416092511862](image-20250416092511862.png)

### A21 双指针区间合并

![image-20250416092600474](image-20250416092600474.png)

### A22 堆序列合并

![image-20250416092640974](image-20250416092640974.png)

### A24 贪心

![image-20250416092717112](image-20250416092717112.png)

### A29 线段覆盖

![image-20250416092743151](image-20250416092743151.png)

### A41 三分小数

**题目描述**

给定 $n$ 个二次函数 $f_1(x),f_2(x),\dots,f_n(x)$（均形如 $ax^2+bx+c$），设 $F(x)=\max\{f_1(x),f_2(x),...,f_n(x)\}$，求 $F(x)$ 在区间 $[0,1000]$ 上的最小值。

**输入格式**

输入第一行为正整数 $T$，表示有 $T$ 组数据。

每组数据第一行一个正整数 $n$，接着 $n$ 行，每行 $3$ 个整数 $a,b,c$，用来表示每个二次函数的 $3$ 个系数，注意二次函数有可能退化成一次。

**输出格式**

每组数据输出一行，表示 $F(x)$ 的在区间 $[0,1000]$ 上的最小值。答案精确到小数点后四位，四舍五入。

输入 #1

```
2
1
2 0 0
2
2 0 0
2 -4 2
```

输出 #1

```
0.0000
0.5000
```

对于 $100\%$ 的数据，$T<10$，$\ n\le 10^4$，$0\le a\le 100$，$|b| \le 5\times 10^3$，$|c| \le 5\times 10^3$。

### A42 三分整数

![image-20250519104503328](image-20250519104503328.png)

![image-20250519104536742](image-20250519104536742.png)

------

## B-搜索算法

### B6 DFS深搜

![image-20250416092841158](image-20250416092841158.png)

![image-20250416092956833](image-20250416092956833.png)

### B15 BFS广搜

![image-20250416093032988](image-20250416093032988.png)

## C-数据结构

### C1 并查集

**题目描述**

如题，现在有一个并查集，你需要完成合并和查询操作。

**输入格式**

第一行包含两个整数 $N,M$ ,表示共有 $N$ 个元素和 $M$ 个操作。

接下来 $M$ 行，每行包含三个整数 $Z_i,X_i,Y_i$ 。

当 $Z_i=1$ 时，将 $X_i$ 与 $Y_i$ 所在的集合合并。

当 $Z_i=2$ 时，输出 $X_i$ 与 $Y_i$ 是否在同一集合内，是的输出 
 `Y` ；否则输出 `N` 。

**输出格式**

对于每一个 $Z_i=2$ 的操作，都有一行输出，每行包含一个大写字母，为 `Y` 或者 `N` 。

输入 #1

```
4 7
2 1 2
1 1 2
2 1 2
1 3 4
2 1 4
1 2 3
2 1 4
```

输出 #1

```
N
Y
N
Y
```

对于 $100\%$ 的数据，$1\le N\le 2\times 10^5$，$1\le M\le 10^6$，$1 \le X_i, Y_i \le N$，$Z_i \in \{ 1, 2 \}$。

**解析：**

这个应用非常广泛，上述题目的正解的时间复杂度为$$O(n)$$，以下为解释：

![image-20250417165010912](image-20250417165010912.png)

![image-20250417165040782](image-20250417165040782.png)

![image-20250417165059357](image-20250417165059357.png)

### C2 线段树

**题目描述**

如题，已知一个数列 $\{a_i\}$，你需要进行下面两种操作：

1. 将某区间每一个数加上 $k$。
2. 求出某区间每一个数的和。

**输入格式**

第一行包含两个整数 $n, m$，分别表示该数列数字的个数和操作的总个数。

第二行包含 $n$ 个用空格分隔的整数 $a_i$，其中第 $i$ 个数字表示数列第 $i$ 项的初始值。

接下来 $m$ 行每行包含 $3$ 或 $4$ 个整数，表示一个操作，具体如下：

1. `1 x y k`：将区间 $[x, y]$ 内每个数加上 $k$。
2. `2 x y`：输出区间 $[x, y]$ 内每个数的和。

**输出格式**

输出包含若干行整数，即为所有操作 2 的结果。

输入 #1

```
5 5
1 5 4 2 3
2 2 4
1 2 3 2
2 3 4
1 1 5 1
2 1 4
```

输出 #1

```
11
8
20
```


对于 $100\%$ 的数据：$1 \le n, m \le {10}^5$，$a_i,k$ 为正数，且任意时刻数列的和不超过 $2\times 10^{18}$。

**【样例解释】**

![](https://cdn.luogu.com.cn/upload/pic/2251.png)

**解析：**

![image-20250417171455028](image-20250417171455028.png)

![image-20250417171654545](image-20250417171654545.png)

![image-20250417171714474](image-20250417171714474.png)

### C5 普通平衡树

**题目描述**

您需要动态地维护一个可重集合 $M$，并且提供以下操作：

1. 向 $M$ 中插入一个数 $x$。
2. 从 $M$ 中删除一个数 $x$（若有多个相同的数，应只删除一个）。
3. 查询 $M$ 中有多少个数比 $x$ 小，并且将得到的答案加一。
4. 查询如果将 $M$ 从小到大排列后，排名位于第 $x$ 位的数。
5. 查询 $M$ 中 $x$ 的前驱（前驱定义为小于 $x$，且最大的数）。
6. 查询 $M$ 中 $x$ 的后继（后继定义为大于 $x$，且最小的数）。

对于操作 3,5,6，**不保证**当前可重集中存在数 $x$。

**输入格式**

第一行为 $n$，表示操作的个数,下面 $n$ 行每行有两个数 $\text{opt}$ 和 $x$，$\text{opt}$ 表示操作的序号（$ 1 \leq \text{opt} \leq 6 $）

**输出格式**

对于操作 $3,4,5,6$ 每行输出一个数，表示对应答案。

输入 #1

```
10
1 106465
4 1
1 317721
1 460929
1 644985
1 84185
1 89851
6 81968
1 492737
5 493598
```

输出 #1

```
106465
84185
492737
```

【数据范围】
对于 $100\%$ 的数据，$1\le n \le 10^5$，$|x| \le 10^7$

### C8 可持续化线段树

**题目背景**

这是个非常经典的可持久化权值线段树入门题——静态区间第 $k$ 小。

**题目描述**

如题，给定 $n$ 个整数构成的序列 $a$，将对于指定的闭区间 $[l, r]$ 查询其区间内的第 $k$ 小值。

**输入格式**

第一行包含两个整数，分别表示序列的长度 $n$ 和查询的个数 $m$。  
第二行包含 $n$ 个整数，第 $i$ 个整数表示序列的第 $i$ 个元素 $a_i$。   
接下来 $m$ 行每行包含三个整数 $ l, r, k$ , 表示查询区间 $[l, r]$ 内的第 $k$ 小值。

**输出格式**

对于每次询问，输出一行一个整数表示答案。

输入 #1

```
5 5
25957 6405 15770 26287 26465 
2 2 1
3 4 1
4 5 1
1 2 2
4 4 1
```

输出 #1

```
6405
15770
26287
25957
26287
```

- 对于 $100\%$ 的数据，满足 $1 \leq n,m \leq 2\times 10^5$，$0\le a_i \leq 10^9$，$1 \leq l \leq r \leq n$，$1 \leq k \leq r - l + 1$。

解析：时间复杂度：$$O(nlogn)$$，$$logn$$查询，原理不解释，毕竟EH看不懂。

### C19 KD树

**题目描述**

给定平面上 $n$ 个点，找出其中的一对点的距离，使得在这 $n$ 个点的所有点对中，该距离为所有点对中最小的

**输入格式**

第一行：$n$ ，保证 $2\le n\le 200000$ 。

接下来 $n$ 行：每行两个实数：$x\ y$ ，表示一个点的行坐标和列坐标，中间用一个空格隔开。

**输出格式**

仅一行，一个实数，表示最短距离，精确到小数点后面 $4$ 位。

**输入输出样例 #1**

输入 #1

```
3
1 1
1 2
2 2
```

输出 #1

```
1.0000
```

数据保证 $0\le x,y\le 10^9$

### C111 莫队

**题目描述**

小B 有一个长为 $n$ 的整数序列 $a$，值域为 $[1,k]$。  他一共有 $m$ 个询问，每个询问给定一个区间 $[l,r]$，求：  $$\sum\limits_{i=1}^k c_i^2$$

其中 $c_i$ 表示数字 $i$ 在 $[l,r]$ 中的出现次数。  小B请你帮助他回答询问。

**输入格式**

第一行三个整数 $n,m,k$。

第二行 $n$ 个整数，表示 小B 的序列。

接下来的 $m$ 行，每行两个整数 $l,r$。

**输出格式**

输出 $m$ 行，每行一个整数，对应一个询问的答案。

输入 #1

```
6 4 3
1 3 2 1 1 3
1 4
2 6
3 5
5 6
```

输出 #1

```
6
9
5
2
```

对于 $100\%$ 的数据，$1\le n,m,k \le 5\times 10^4$。

思路：

![image-20250519115428164](image-20250519115428164.png)

不证明了，无所谓。

-------

## D 图论



### D2 狄克斯特拉算法

**题目描述**

如题，给出一个有向图，请输出从某一点出发到所有点的最短路径长度。

**输入格式**

第一行包含三个整数 $n,m,s$，分别表示点的个数、有向边的个数、出发点的编号。

接下来 $m$ 行每行包含三个整数 $u,v,w$，表示一条 $u \to v$ 的，长度为 $w$ 的边。

**输出格式**

输出一行 $n$ 个整数，第 $i$ 个表示 $s$ 到第 $i$ 个点的最短路径，若不能到达则输出 $2^{31}-1$。

输入 #1

```
4 6 1
1 2 2
2 3 2
2 4 1
1 3 5
3 4 3
1 4 4
```

输出 #1

```
0 2 4 3
```

对于 $100\%$ 的数据：$1 \le n \le 10^4$，$1\le m \le 5\times 10^5$，$1\le u,v\le n$，$w\ge 0$，$\sum w< 2^{31}$，保证数据随机。

**解析：**时间复杂度：$$O(n^2)$$

![image-20250418090828454](image-20250418090828454.png)

**堆优化版：**

**题目描述**

给定一个 $n$ 个点，$m$ 条有向边的带非负权图，请你计算从 $s$ 出发，到每个点的距离。

数据保证你能从 $s$ 出发到任意点。

**输入格式**

第一行为三个正整数 $n, m, s$。
第二行起 $m$ 行，每行三个非负整数 $u_i, v_i, w_i$，表示从 $u_i$ 到 $v_i$ 有一条权值为 $w_i$ 的有向边。

**输出格式**

输出一行 $n$ 个空格分隔的非负整数，表示 $s$ 到每个点的距离。

输入 #1

```
4 6 1
1 2 2
2 3 2
2 4 1
1 3 5
3 4 3
1 4 4
```

输出 #1

```
0 2 4 3
```

$1 \leq n \leq 10^5$；

$1 \leq m \leq 2\times 10^5$；

$s = 1$；

$1 \leq u_i, v_i\leq n$；

$0 \leq w_i \leq 10 ^ 9$,

$0 \leq \sum w_i \leq 10 ^ 9$。

![image-20250418091036262](image-20250418091036262.png)

### D4 最短路 Floyd 算法

![image-20250418091227375](image-20250418091227375.png)

### D5 全源 Johnson 算法

**题目描述**

给定一个包含 $n$ 个结点和 $m$ 条带权边的有向图，求所有点对间的最短路径长度，一条路径的长度定义为这条路径上所有边的权值和。

注意：边权**可能**为负，且图中**可能**存在重边和自环；

**输入格式**

第 $1$ 行：$2$ 个整数 $n,m$，表示给定有向图的结点数量和有向边数量。

接下来 $m$ 行：每行 $3$ 个整数 $u,v,w$，表示有一条权值为 $w$ 的有向边从编号为 $u$ 的结点连向编号为 $v$ 的结点。

**输出格式**

若图中存在负环，输出仅一行 $-1$。

若图中不存在负环：

输出 $n$ 行：令 $dis_{i,j}$ 为从 $i$ 到 $j$ 的最短路，在第 $i$ 行输出 $\sum\limits_{j=1}^n j\times dis_{i,j}$，注意这个结果可能超过 int 存储范围。

如果不存在从 $i$ 到 $j$ 的路径，则 $dis_{i,j}=10^9$；如果 $i=j$，则 $dis_{i,j}=0$。

**输入输出样例 #1**

输入 #1

```
5 7
1 2 4
1 4 10
2 3 7
4 5 3
4 2 -2
3 4 -3
5 3 4
```

输出 #1

```
128
1000000072
999999978
1000000026
1000000014
```

**输入输出样例 #2**

输入 #2

```
5 5
1 2 4
3 4 9
3 4 -3
4 5 3
5 3 -2
```

输出 #2

```
-1
```

对于 $100\%$ 的数据，$1\leq n\leq 3\times 10^3,\ \ 1\leq m\leq 6\times 10^3,\ \ 1\leq u,v\leq n,\ \ -3\times 10^5\leq w\leq 3\times 10^5$。

**解析：**EH不会，只是默默地放图片说明一切。

![image-20250418091441310](image-20250418091441310.png)

### D7、D8 Prim、Kruscal算法-最小生成树

**题目描述**

如题，给出一个无向图，求出最小生成树，如果该图不连通，则输出 `orz`。

**输入格式**

第一行包含两个整数 $N,M$，表示该图共有 $N$ 个结点和 $M$ 条无向边。

接下来 $M$ 行每行包含三个整数 $X_i,Y_i,Z_i$，表示有一条长度为 $Z_i$ 的无向边连接结点 $X_i,Y_i$。

**输出格式**

如果该图连通，则输出一个整数表示最小生成树的各边的长度之和。如果该图不连通则输出 `orz`。

**输入输出样例 #1**

输入 #1

```
4 5
1 2 2
1 3 2
1 4 3
2 3 4
3 4 3
```

输出 #1

```
7
```

对于 $100\%$ 的数据：$1\le N\le 5000$，$1\le M\le 2\times 10^5$，$1\le Z_i \le 10^4$。


样例解释：

 ![](https://cdn.luogu.com.cn/upload/pic/2259.png) 

所以最小生成树的总边权为 $2+2+3=7$。

**解析：**

Prim算法（$$O(n^2)$$）：

![image-20250418091937605](image-20250418091937605.png)

Kruscal算法（$$O(mlogm)$$）：

![image-20250418092200766](image-20250418092200766.png)

### D11 树链剖分（最近公共祖先 LCA）

**题目描述**

如题，给定一棵有根多叉树，请求出指定两个点直接最近的公共祖先。

**输入格式**

第一行包含三个正整数 $N,M,S$，分别表示树的结点个数、询问的个数和树根结点的序号。

接下来 $N-1$ 行每行包含两个正整数 $x, y$，表示 $x$ 结点和 $y$ 结点之间有一条直接连接的边（数据保证可以构成树）。

接下来 $M$ 行每行包含两个正整数 $a, b$，表示询问 $a$ 结点和 $b$ 结点的最近公共祖先。

**输出格式**

输出包含 $M$ 行，每行包含一个正整数，依次为每一个询问的结果。

**输入输出样例 #1**

输入 #1

```
5 5 4
3 1
2 4
5 1
1 4
2 4
3 2
3 5
1 2
4 5
```

输出 #1

```
4
4
1
4
4
```

对于 $100\%$ 的数据，$1 \leq N,M\leq 500000$，$1 \leq x, y,a ,b \leq N$，**不保证** $a \neq b$。


样例说明：

该树结构如下：

 ![](https://cdn.luogu.com.cn/upload/pic/2282.png) 

第一次询问：$2, 4$ 的最近公共祖先，故为 $4$。

第二次询问：$3, 2$ 的最近公共祖先，故为 $4$。

第三次询问：$3, 5$ 的最近公共祖先，故为 $1$。

第四次询问：$1, 2$ 的最近公共祖先，故为 $4$。

第五次询问：$4, 5$ 的最近公共祖先，故为 $4$。

故输出依次为 $4, 4, 1, 4, 4$。

**解析：**

![image-20250418092432087](image-20250418092432087.png)

### D13 树上距离

![image-20250418092643205](image-20250418092643205.png)

解析：无，LCA板子题举例而已。

------

## E-动态规划

### E4 最长上升子序列（二分优化）

**题目描述**

给出一个由 $n(n\le 5000)$ 个不超过 $10^6$ 的正整数组成的序列。请输出这个序列的**最长上升子序列**的长度。

最长上升子序列是指，从原序列中**按顺序**取出一些数字排在一起，这些数字是**逐渐增大**的。

**输入格式**

第一行，一个整数 $n$，表示序列长度。

第二行有 $n$ 个整数，表示这个序列。

**输出格式**

一个整数表示答案。

**输入输出样例 #1**

输入 #1

```
6
1 2 4 1 3 4
```

输出 #1

```
4
```

**解析：**时间复杂度：$$O(nlogn)$$

![image-20250418140921987](image-20250418140921987.png)

### E5 最长公共子序列

**题目描述**

给出一个长度为$n(n\le 5000)$ 的字符串序列。请输出这个序列的**最长公共子序列**及其长度。

**输入格式**

输入两个字符串（代码里忽略了这个）。

**输出格式**

输出分为两行，

第一行输出一个整数表示最长公共子序列的长度，

第二行输出这个最长公共子序列。

**解析：**时间复杂度：$$O(n^2)$$

![image-20250418141328421](image-20250418141328421.png)

**p数组记录转移方向，用来还原子序列。**

![image-20250418141439935](image-20250418141439935.png)

### E6 最长公共子串

**题目描述**

给出一个长度为$n(n\le 5000)$ 的字符串序列。请输出这个序列的**最长公共子串**及其长度。

**输入格式**

输入两个字符串（代码里忽略了这个）。

**输出格式**

输出一个整数表示最长公共子串的长度，

**解析：**时间复杂度：$$O(n^2)$$

![image-20250418142032244](image-20250418142032244.png)

![image-20250418142022435](image-20250418142022435.png)

### E8 01背包

**题目描述**

有 $N$ 件物品和一个容量为 $M$ 的背包。第 $i$ 件物品的重量是 $W_i$，价值是 $D_i$。求解将哪些物品装入背包可使这些物品的重量总和不超过背包容量，且价值总和最大。

**输入格式**

第一行：物品个数 $N$ 和背包大小 $M$。

第二行至第 $N+1$ 行：第 $i$ 个物品的重量 $W_i$ 和价值 $D_i$。

**输出格式**

输出一行最大价值。

**输入输出样例 #1**

输入 #1

```
4 6
1 4
2 6
3 12
2 7
```

输出 #1

```
23
```

**解析：**时间复杂度：$$O(mn)$$，空间复杂度方面，可以使用一维数组+逆向查找优化。

![image-20250418142142298](image-20250418142142298.png)



### E9 完全背包

题目不码了，跟上题一样，只不过是把只能拿一件物品改为可以拿任意件。

**解析：**时间复杂度：$$O(mn)$$，空间复杂度方面，可以使用一维数组优化。

![image-20250418142653554](image-20250418142653554.png)



### E10 多重背包

**题目描述**

小 FF 对洞穴里的宝物进行了整理，他发现每样宝物都有一件或者多件。他粗略估算了下每样宝物的价值，之后开始了宝物筛选工作：小 FF 有一个最大载重为 $W$ 的采集车，洞穴里总共有 $n$ 种宝物，每种宝物的价值为 $v_i$，重量为 $w_i$，每种宝物有 $m_i$ 件。小 FF 希望在采集车不超载的前提下，选择一些宝物装进采集车，使得它们的价值和最大。

**输入格式**

第一行为两个整数 $n$ 和 $W$，分别表示宝物种数和采集车的最大载重。

接下来 $n$ 行每行三个整数 $v_i,w_i,m_i$。

**输出格式**

输出仅一个整数，表示在采集车不超载的情况下收集的宝物的最大价值。

**输入输出样例 #1**

输入 #1

```
4 20
3 9 3
5 9 1
9 4 2
8 1 3
```

输出 #1

```
47
```

对于 $100\%$ 的数据，$n\leq \sum m_i \leq 10^5$，$0\le W\leq 4\times 10^4$，$1\leq n\le 100$。

![image-20250418142914118](image-20250418142914118.png)

### E11 滑动窗口

**题目描述**

有一个长为 $n$ 的序列 $a$，以及一个大小为 $k$ 的窗口。现在这个从左边开始向右滑动，每次滑动一个单位，求出每次滑动后窗口中的最大值和最小值。

例如，对于序列 $[1,3,-1,-3,5,3,6,7]$ 以及 $k = 3$，有如下过程：

$$\def\arraystretch{1.2}
\begin{array}{|c|c|c|}\hline
\textsf{窗口位置} & \textsf{最小值} & \textsf{最大值} \\ \hline
\verb![1   3  -1] -3   5   3   6   7 ! & -1 & 3 \\ \hline
\verb! 1  [3  -1  -3]  5   3   6   7 ! & -3 & 3 \\ \hline
\verb! 1   3 [-1  -3   5]  3   6   7 ! & -3 & 5 \\ \hline
\verb! 1   3  -1 [-3   5   3]  6   7 ! & -3 & 5 \\ \hline
\verb! 1   3  -1  -3  [5   3   6]  7 ! & 3 & 6 \\ \hline
\verb! 1   3  -1  -3   5  [3   6   7]! & 3 & 7 \\ \hline
\end{array}
$$

**输入格式**

输入一共有两行，第一行有两个正整数 $n,k$。
第二行 $n$ 个整数，表示序列 $a$

**输出格式**

输出共两行，第一行为每次窗口滑动的最小值   
第二行为每次窗口滑动的最大值

**输入输出样例 #1**

输入 #1

```
8 3
1 3 -1 -3 5 3 6 7
```

输出 #1

```
-1 -3 -3 -3 3 3
3 3 5 5 6 7
```

   对于 $100\%$ 的数据，$1\le k \le n \le 10^6$，$a_i \in [-2^{31},2^{31})$。

时间复杂度：$$O(n)$$

![image-20250418143122329](image-20250418143122329.png)

### E17 树形DP

**EH声明：代码使用了链表储存结构，感觉不是很符合个人习惯，可以酌情更换方法。**

**题目描述**

某大学有 $n$ 个职员，编号为 $1\ldots n$。

他们之间有从属关系，也就是说他们的关系就像一棵以校长为根的树，父结点就是子结点的直接上司。

现在有个周年庆宴会，宴会每邀请来一个职员都会增加一定的快乐指数 $r_i$，但是呢，如果某个职员的直接上司来参加舞会了，那么这个职员就无论如何也不肯来参加舞会了。

所以，请你编程计算，邀请哪些职员可以使快乐指数最大，求最大的快乐指数。

**输入格式**

输入的第一行是一个整数 $n$。

第 $2$ 到第 $(n + 1)$ 行，每行一个整数，第 $(i+1)$ 行的整数表示 $i$ 号职员的快乐指数 $r_i$。

第 $(n + 2)$ 到第 $2n$ 行，每行输入一对整数 $l, k$，代表 $k$ 是 $l$ 的直接上司。

**输出格式**

输出一行一个整数代表最大的快乐指数。

**输入输出样例 #1**

输入 #1

```
7
1
1
1
1
1
1
1
1 3
2 3
6 4
7 4
4 5
3 5
```

输出 #1

```
5
```

对于 $100\%$ 的数据，保证 $1\leq n \leq 6 \times 10^3$，$-128 \leq r_i\leq 127$，$1 \leq l, k \leq n$，且给出的关系一定是一棵树。

**解析：**

![image-20250418143319059](image-20250418143319059.png)



------

## F-字符串

EH过度劳累，此时的san值有点低，精神奔溃中，写水一点吧。

### F3 KMP算法

**题目描述**

给出两个字符串 $s_1$ 和 $s_2$，若 $s_1$ 的区间 $[l, r]$ 子串与 $s_2$ 完全相同，则称 $s_2$ 在 $s_1$ 中出现了，其出现位置为 $l$。  
现在请你求出 $s_2$ 在 $s_1$ 中所有出现的位置。

定义一个字符串 $s$ 的 border 为 $s$ 的一个**非 $s$ 本身**的子串 $t$，满足 $t$ 既是 $s$ 的前缀，又是 $s$ 的后缀。  
对于 $s_2$，你还需要求出对于其每个前缀 $s'$ 的最长 border $t'$ 的长度。

**输入格式**

第一行为一个字符串，即为 $s_1$。  
第二行为一个字符串，即为 $s_2$。

**输出格式**

首先输出若干行，每行一个整数，**按从小到大的顺序**输出 $s_2$ 在 $s_1$ 中出现的位置。  
最后一行输出 $|s_2|$ 个整数，第 $i$ 个整数表示 $s_2$ 的长度为 $i$ 的前缀的最长 border 长度。

**输入输出样例 #1**

输入 #1

```
ABABABC
ABA
```

输出 #1

```
1
3
0 0 1
```

样例 1 解释

 ![](https://cdn.luogu.com.cn/upload/pic/2257.png)。

对于 $s_2$ 长度为 $3$ 的前缀 `ABA`，字符串 `A` 既是其后缀也是其前缀，且是最长的，因此最长 border 长度为 $1$。

对于全部的测试点，保证 $1 \leq |s_1|,|s_2| \leq 10^6$，$s_1, s_2$ 中均只含大写英文字母。

**解析：**

![image-20250418154051216](image-20250418154051216.png)

**难点主要是next的计算：**

![image-20250418154128824](image-20250418154128824.png)



### F6 Trie字典树

代码对应的题目：

**输入I表示插入，否则是查询，查询时输出即可：**

![image-20250418154637702](image-20250418154637702.png)

**建树过程：**

![image-20250418154444650](image-20250418154444650.png)

### F7 最大异或对

![image-20250418154901774](image-20250418154901774.png)

**EH是懒鬼，这段代码没有测验，直接AI写的，管他呢。。。**

------

## G-数学

### G1 快速幂

**题目描述**

给你三个整数 $a,b,p$，求 $a^b \bmod p$。

**输入格式**

输入只有一行三个整数，分别代表 $a,b,p$。

**输出格式**

输出一行一个字符串 `a^b mod p=s`，其中 $a,b,p$ 分别为题目给定的值， $s$ 为运算结果。

**输入输出样例 #1**

输入 #1

```
2 10 9
```

输出 #1

```
2^10 mod 9=7
```

对于 $100\%$ 的数据，保证 $0\le a,b < 2^{31}$，$a+b>0$，$2 \leq p \lt 2^{31}$。

### G5 gcd及lcm问题

**题目描述**

输入两个正整数 $x_0, y_0$，求出满足下列条件的 $P, Q$ 的个数：

1. $P,Q$ 是正整数。

2. 要求 $P, Q$ 以 $x_0$ 为最大公约数，以 $y_0$ 为最小公倍数。

试求：满足条件的所有可能的 $P, Q$ 的个数。

**输入格式**

一行两个正整数 $x_0, y_0$。

**输出格式**

一行一个数，表示求出满足条件的 $P, Q$ 的个数。

**输入输出样例 #1**

输入 #1

```
3 60
```

输出 #1

```
4
```

对于 $100\%$ 的数据，$2 \le x_0, y_0 \le {10}^5$。

### G8 线性筛质数

**题目描述**

如题，给定一个范围 $n$，有 $q$ 个询问，每次输出第 $k$ 小的素数。

**输入格式**

第一行包含两个正整数 $n,q$，分别表示查询的范围和查询的个数。

接下来 $q$ 行每行一个正整数 $k$，表示查询第 $k$ 小的素数。

**输出格式**

输出 $q$ 行，每行一个正整数表示答案。

**输入输出样例 #1**

输入 #1

```
100 5
1
2
3
4
5
```

输出 #1

```
2
3
5
7
11
```

对于 $100\%$ 的数据，$n = 10^8$，$1 \le q \le 10^6$，保证查询的素数不大于 $n$。

**解析：**时间复杂度：$$O(n)$$

![image-20250418170238198](image-20250418170238198.png)



### G13 费马小定理

没啥好说的，直接记结论得了。

![image-20250418170352053](image-20250418170352053.png)

### G49 向量运算

![image-20250517204844570](image-20250517204844570.png)

### G50 线线关系

![image-20250517205357054](image-20250517205357054.png)

### G52 凸包算法

没有py代码，因为EH还不会。

**题目描述**

农夫约翰想要建造一个围栏用来围住他的奶牛，可是他资金匮乏。他建造的围栏必须包括他的奶牛喜欢吃草的所有地点。对于给出的这些地点的坐标，计算最短的能够围住这些点的围栏的长度。

**输入格式**

输入数据的第一行是一个整数。表示农夫约翰想要围住的放牧点的数目 $n$。

第 $2$ 到第 $(n + 1)$ 行，每行两个实数，第 $(i + 1)$ 行的实数 $x_i, y_i$ 分别代表第 $i$ 个放牧点的横纵坐标。

**输出格式**

输出输出一行一个四舍五入保留两位小数的实数，代表围栏的长度。

**输入输出样例 #1**

输入 #1

```
4
4 8
4 12
5 9.3
7 8
```

输出 #1

```
12.00
```

对于 $100\%$ 的数据，保证 $1 \leq n \leq 10^5$，$-10^6 \leq x_i, y_i \leq 10^6$。小数点后最多有 $2$ 位数字。

### G53 旋转卡壳

**题目描述**

给定平面上 $n$ 个点，求凸包直径。

**输入格式**

第一行一个正整数 $n$。   
接下来 $n$ 行，每行两个整数 $x,y$，表示一个点的坐标。保证所有点的坐标两两不同。

**输出格式**

输出一行一个整数，表示答案的平方。

**输入输出样例 #1**

输入 #1

```
4
0 0
0 1
1 1
1 0
```

输出 #1

```
2
```

对于 $100\%$ 的数据，$2\le n \le 50000$，$|x|,|y| \le 10^4$。

