---
title: Permutations II
date: 2020-07-12 10:24
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

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

For example,
`[1,1,2]` have the following unique permutations:

```c++
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

## Analysis

还是回溯问题。这是之前的[Permutations]({% post_url 2020-07-12-permutations%})的延伸。这里进一步给出了有重复数字的数组，要求解没有重复。简单的可以用set的唯一性。这里想要一种直接的方法把相同的解过滤掉。这就是针对相同的数字，怎么来处理。也是参考了各种解法，这里用visited的数组，进一步判断，要求对于相同的数字，之前的数字必须访问过之后，才可以访问后面的数字。比如这里有两个 1， 我们记为 `1, 1'`, 这样的话我们只有`1, 1'`是一个合适的解，而`1', 1`则不是，也就是说要求相同的数字的时候，规定一个顺序，只有复合一种顺序的是一个解。所以代码里进一步判断，只有当前面的1被访问过之后，才算是一个解，否则就continue

## Solution

```c++
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> visited(nums.size(), 0);
        vector<int> out;
        sort(nums.begin(), nums.end());
        dfs(nums, visited, res, out);
        return res;
    }
    
    void dfs(const vector<int> &nums, vector<int> &visited, vector<vector<int>> &res, vector<int> &out) {
        if (out.size() == nums.size()) {
            res.push_back(out);
        } else {
            for (int i = 0; i < nums.size(); ++i) {
                if (visited[i] == 0) {
                    if (i > 0 && nums[i] == nums[i-1] && visited[i-1] == 0) continue;
                    visited[i] = 1;
                    out.push_back(nums[i]);
                    dfs(nums, visited, res, out);
                    out.pop_back();
                    visited[i] = 0;
                }
            }
        }
    }
};
```

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        visited = [0] * len(nums)
        nums.sort()
        self.dfs(nums, visited, res, [])
        return res
    
    def dfs(self, nums, visited, res, out):
        if len(out) == len(nums):
            res.append(out)
        else:
            for i in range(len(nums)):
                if visited[i] == 0: 
                    if i>0 and nums[i] == nums[i-1] and visited[i-1] == 0:
                        continue

                    visited[i] = 1
                    self.dfs(nums, visited, res, out +[nums[i]])
                    visited[i] = 0
```
