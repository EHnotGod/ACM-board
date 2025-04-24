#include<bits/stdc++.h>
using namespace std;

int n,m;
string s[100005],s1;
map<string,bool> word;
map<string,int> cnt;
int sum,len;
//s[] 记录文章中的单词
//word[] 记录单词表中的单词
//cnt[] 记录文章当前区间内单词出现次数
//sum 记录文章中出现单词表的单词数
//len 记录包含表中单词最多的区间的最短长度

int main(){
  cin>>n;
  for(int i=1;i<=n;i++)cin>>s1,word[s1]=1;
  cin>>m;
  for(int j=1,i=1; j<=m; j++){  //i<=j
    cin>>s[j];
    if(word[s[j]]) cnt[s[j]]++;
    if(cnt[s[j]]==1) sum++, len=j-i+1;
    while(i<=j){
      if(cnt[s[i]]==1) break; //保持i指针位置不动
      if(cnt[s[i]]>=2) cnt[s[i]]--,i++; //去重,更优
      if(!word[s[i]]) i++; //去掉非目标单词,更优
    }
    len=min(len,j-i+1); //更新
  }
  cout<<sum<<endl<<len<<endl;
}