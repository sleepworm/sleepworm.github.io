---
title: Subsets II
date: 2020-06-27 14:58
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

Given a collection of integers that might contain duplicates, **nums**, return all possible subsets.

**Note:** The solution set must not contain duplicate subsets.

For example,
If **nums** = `[1,2,2]`, a solution is:

```c++
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

## Analysis
典型的回溯。这是之前[Subsets]({% post_url 2020-06-27-subsets%})的延伸，对于直接解题的方案，每次循环元素的时候，把后面重复的忽略掉就可以了。可以分析一下例子[[]] —> [[],[1]] —> [[], [1], [2], [1,2]] —> [[], [1], [2], [1,2], [2,2], [1,2,2]]。 这里可以看出对于重复的元素，只在后来添加的元素上加上这个2. 即只在 [2], [1,2] 后面加上2. 所以这里对于一个新来的元素需要判断，如果相同的话，那么对于集合的起始点，是上一次添加元素之前的size.

对于回溯的方法解题， 这里也是需要合理的忽略掉重复的元素

## Solution

### 记录所有可得到的集合
```c++
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<int> ss;
        vector<vector<int> > res;
        subsetsN(nums, 0, ss, res);
        return res;
    }
    
    void subsetsN(vector<int> & nums,  int pos, vector<int> &ss, vector<vector<int> > &res) {
        res.push_back(ss); // push every set obtained
        for (int i=pos; i < nums.size(); ++i) {
            if (i > pos && nums[i] == nums[i-1]) continue;
            ss.push_back(nums[i]);
            subsetsN(nums, i+1, ss, res);
            ss.pop_back();
        }
    }
};
```

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        self.dfs(nums, 0, res, [])
        return res
    
    def dfs(self, nums, pos, res, out):
        res.append(out)
        for i in range (pos, len(nums)):
            if i > pos and nums[i] == nums[i-1]:
                pass
            else:
                self.dfs(nums, i+1, res, out + [nums[i]]
```

### 循环直接解题
```c++
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>>res(1);
        int pre_len = 1;
        for (int i = 0; i < nums.size(); ++i) {
            int start_pos = 0;
            if (i > 0 && nums[i] == nums[i-1]) {
                start_pos = pre_len;
            }
            pre_len = res.size();
            for (int j = start_pos; j < pre_len;++j) {
                res.push_back(res[j]);
                res.back().push_back(nums[i]);
            }
        }
        return res;
    }
};
```

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        res.append([])
        start_pos, pre_len = 0, 0
        for i in range(len(nums)):
            start_pos = pre_len if i > 0 and nums[i] == nums[i-1] else 0
            pre_len = len(res)
            for j in range(start_pos, pre_len):
                temp = res[j] + [nums[i]]
                res.append(temp)
 
        return res
```