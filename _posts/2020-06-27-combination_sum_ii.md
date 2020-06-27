---
title: Combination Sum II
date: 2020-06-27 09:50
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

## 1 Question

Given a collection of candidate numbers (**C**) and a target number (**T**), find all unique combinations in **C** where the candidate numbers sums to **T**.

Each number in **C** may only be used **once** in the combination.

**Note:**

- All numbers (including target) will be positive integers.
- The solution set must not contain duplicate combinations.

For example, given candidate set `[10, 1, 2, 7, 6, 1, 5]` and target `8`, 
A solution set is: 

```c++
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

## Analysis
这道题是对之前[Combination Sum]({% post_url 2020-06-20-combination_sum %})的一个变形，这里有每个数字只能用一次的限制。解题上来看，还是一样的。只是需要考虑怎么来处理只用一次，之前我们每次内循环都还是从原来的数字index开始，这里我们只要每次用完一个数字就移到下一个数字就可以了。代码如下

## Solution

```c++
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<int> out;
        vector<vector<int> > res;
        sort(candidates.begin(), candidates.end());
        dfs(candidates, target, 0, out, res);
        return res;
    }

    void dfs(const vector<int> &candidates, int &target, int level, vector<int> &out, vector<vector<int> > &res) {
        if(target == 0) res.push_back(out);
        else {
            for (int i=level; i<candidates.size(); ++i) {
                if (i > level && candidates[i] == candidates[i-1]) continue;
                if(target >= candidates[i]) {
                    out.push_back(candidates[i]);
                    target -= candidates[i];
                    dfs(candidates, target, i+1, out, res); // 这里用i+1的方式保证当前的元素只用一次
                    target += candidates[i];
                    out.pop_back();
                }
            }
        }
    }
};
```

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        res = []
        self.dfs(candidates, target, 0, res, [])
        return res

    def dfs(self, candidates, target, level, res, out):
        if target == 0:
            res.append(copy.deepcopy(out))
            # 如果下面用out + [candidates[i]]， 这里可以
            # res.append(out)
            return
        for i in range(level, len(candidates)):
            ## here i> level don't i > 0
            if i>level and candidates[i] == candidates[i-1]:
                continue
            if target >= candidates[i]:
                out.append(candidates[i])
                self.dfs(candidates, target-candidates[i], i+1, res, out)
                out.pop()
                # 上面三行也可以替换成
                # self.dfs(candidates, target-candidates[i], i+1, res, out + [candidates[i]])
```