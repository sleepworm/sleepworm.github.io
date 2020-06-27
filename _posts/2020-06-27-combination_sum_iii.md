---
title: Combination Sum III
date: 2020-06-27 10:10
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

Find all possible combinations of ***k*** numbers that add up to a number ***n***, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

***Example 1:***

Input: ***k*** = 3, ***n*** = 7

Output:

```c++
[[1,2,4]]
```

***Example 2:***

Input: ***k*** = 3, ***n*** = 9

Output:

```c++
[[1,2,6], [1,3,5], [2,3,4]]
```


## Analysis
这道题类似之前的[Combination Sum]({% post_url 2020-06-20-combination_sum %})和[Combination Sum II]({% post_url 2020-06-27-combination_sum_ii %})。注意这里的题目要求，(题目好像并没有要求每个元素只能用一次，但是给出的例子是每个元素只用了一次)， 所以要求： 1，同一个元素可以用1次， 2. 所求集合唯一。所以需要一个level， 每次从当前元素的下一个元素即i+1开始，对于下一次遍历从level=i开始。(如果同一个元素可以用多次，则每次从i开始，即从当前元素开始)

## Solution

```c++
class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int> > res;
        vector<int> out;
        dfs(k, n, 1, out, res);
        return res;
    }
    
    void dfs(int k, int n, int pos, vector<int>&out, vector<vector<int> > &res) {
        if (0 == k && n == 0) res.push_back(out);
        else {
            for (int i=pos; i <=9; ++i) {
                if (i <= n) {
                    out.push_back(i);
                    dfs(k-1, n-i, i+1, out, res); // i+1 元素只用一次
                    out.pop_back();
                }
            }
        }
    }
};
```

```python
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        res = []
        self.dfs(k, 1, res, [], n)
        return res
    
    def dfs(self, k, level, res, out, n):
        if n == 0 and len(out) == k:
            res.append(out)
            return
        for i in range(level, 10):
            if n >= i and len(out) < k:
                self.dfs(k, i+1, res, out + [i], n-i)
```

<!-- ```c++
class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> out;
        dfs(k, n, 1, res, out);
        return res;
    }
    
    void dfs (int k , int n, int start, vector<vector<int> > &res, vector<int> &out) {
        if (k == 0 && n == 0) {
            res.push_back(out);
        } 
        
        for (int i = start; i <= 9; ++i) {
            if (k > 0 && i <= n) {
                out.push_back(i);
                dfs(k-1, n-i, i, res, out);
                out.pop_back();
            }
        }
    }
}; -->
```