---
title: Combinations
date: 2020-07-07 21:34
categories:
- Leetcode
- array
- 数组
- backtracking
- 回溯法
tags:
- Leetcode
- array
- 数组
- backtracking
- 回溯法
---

## Question

Given two integers *n* and *k*, return all possible combinations of *k* numbers out of 1 ... *n*.

For example,
If *n* = 4 and *k* = 2, a solution is:

```c++
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```


## Analysis
典型的回溯问题。解题中遇到的问题对于一个固定的己选值，怎样确定它下面所有可能的解值。比如选定了一个数i， 下面选择的数要从i+1开始。所以对DFS要传进去一个值表示从当前这个值开始做为可能的子解。其它就是典型的回溯了。

## Solution

### 记录所有可得到的集合
```c++
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> out;
        dfs(n, k, 1, res, out);
        return res;
    }

    void dfs(int n, int k, int level, vector<vector<int>> &res, vector<int> &out) {
        if (k == 0) {
            res.push_back(out);
            return;
        }
        for (int i = level; i <= n; ++i) {
            out.push_back(i);
            dfs(n, k-1, i+1, res, out);
            out.pop_back();
        }
    }
};
```

```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        res = []
        self.dfs(n, k, 1, res, [])
        return res

    def dfs(self, n, k, level, res, out):
        if k == 0:
            res.append(out)
        else:
            for i in range(level, n+1):
                self.dfs(n, k-1, i+1, res, out + [i])]
```
