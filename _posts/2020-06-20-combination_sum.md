---
title: Combination Sum
date: 2020-06-20 13:50
categories:
- Leetcode
- array
- 数组
- backtracking
- 回溯法
- learn python
tags:
- Leetcode
- array
- 数组
- backtracking
- 回溯法
- learn python
---

## Question

Given a set of candidate numbers (**C**) and a target number (**T**), find all unique combinations in **C** where the candidate numbers sums to **T**.

The **same** repeated number may be chosen from **C** unlimited number of times.

**Note:**

- All numbers (including target) will be positive integers.
- The solution set must not contain duplicate combinations.

For example, given candidate set `[2, 3, 6, 7]` and target `7`, 
A solution set is: 

```c++
[
  [7],
  [2, 2, 3]
]
```


## Analysis
这道题看起来还是挺麻烦的，要搜索所有的解，对于解题来看，先简单的想一个例子，然后walk through，仔细理理应该可以想出来需要一步步的搜索加上回溯。相对来说还是比较复杂了

## Solution

```c++
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<int> out;
        vector<vector<int> > res;
        dfs(candidates, target, 0, out, res);
        return res;
    }

    void dfs(const vector<int> &candidates, int &target, int level, vector<int> &out, vector<vector<int> > &res) {
        if(target == 0) res.push_back(out);
        else {
            for (int i=level; i<candidates.size(); ++i) {
                if(target >= candidates[i]) {
                    out.push_back(candidates[i]);
                    target -= candidates[i];
                    dfs(candidates, target, i, out, res);
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
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        self.dfs(res, [], candidates, 0, target)
        return res

    def dfs (self, res, out, candidates, level, target):
        if target == 0:
            # 引用调用，最后所有的out都变成[]， 所以这里一定要deepcopy
            res.append(copy.deepcopy(out))
        else:
            for i in range (level, len(candidates)):
                if target >= candidates[i]:
                    out.append(candidates[i])
                    self.dfs(res, out, candidates, i, target-candidates[i])
                    out.pop()
```

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        self.dfs(res, [], candidates, 0, target)
        return res

    def dfs (self, res, out, candidates, level, target):
        if target == 0:
            res.append(out)
        else:
            for i in range (level, len(candidates)):
                if target >= candidates[i]:
                    # out.append(candidates[i])
                    # out +[candidates[i]]实际上是生成了另一个list指向另一个reference，所以事实上一旦放进res之后就不会再touch
                    self.dfs(res, out + [candidates[i]], candidates, i, target-candidates[i])
                    # out.pop()
```

### Call stack

简单的来画一个call stack，帮助理解。假设需要搜索的数组为`[2,3,7]`, target 为 `7`。外层需要有一个循环来不断的找下一个元素，如果满中条件，则找到一个结果，如果不满足条件，则对元素遍历收索一遍，然后再退回来，重新开始一次新的遍历搜索
![backtracking_call_stack](/assets/images/leetcode/39_combination_sum.svg)

## Python 知识点

先看例子

```python
a = [1,2,3]
b = a
a.append(4)
a += [5]
a = a + [6]
print('a', a)
print('b', b)

# output
a [1, 2, 3, 4, 5, 6]
b [1, 2, 3, 4, 5]
```

Expression `a += [5]` modifies the list in-place, means it extends the list such that “list1” and “list2” still have the reference to the same list.

Expression `a = a + [6]` creates a new list and changes “list1” reference to that new list and “list2” still refer to the old list. so here `a = a + [6]` makes a completely a different reference from `b` while previously, `a` and `b` have the reference to the same list.
