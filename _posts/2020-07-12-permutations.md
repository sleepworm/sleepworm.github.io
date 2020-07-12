---
title: Permutations
date: 2020-07-12 09:53
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

Given a collection of **distinct** numbers, return all possible permutations.

For example,
`[1,2,3]` have the following permutations:

```c++
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

## Analysis

还是回溯问题。开始解题的时候没有想明白具体的解，在排序过程中可以任意排序，所以不清楚怎么要进行排序。参考了一些解法。还是一样，每次都从头开始遍历，用一个数组来记录这个数字是不是已经访问过，如果是的话，就不再访问。直到输出的数组长度包括了所有的数字。感觉上还是有一点小小的巧妙的地方的，用一个visited的数组来记录已经访问过的数字。

## Solution

```c++
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> visited(nums.size(), 0);
        vector<vector<int>> res;
        vector<int> out;
        dfs(nums, visited, res, out);
        return res;
    }
    
    void dfs(vector<int> &nums, vector<int> &visited, vector<vector<int>>&res, vector<int> &out) {
        if (out.size() == nums.size()) res.push_back(out);
        else {
            for (int i = 0; i < nums.size(); ++i) {
                if (!visited[i]) {
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
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        # visited = [0 for i in range (len(nums))]
        visited = [0] * len(nums)
        self.dfs (nums, visited, res, [])
        return res
    
    def dfs(self, nums, visited, res, out):
        if len(out) == len(nums):
            res.append(out)
        else :
            for i in range (len(nums)):
                if visited[i] == 0:
                    visited[i] = 1
                    self.dfs(nums, visited, res, out + [nums[i]])
                    visited[i] = 0
```
