---
title: Two Sum II Input Array Is Sorted
date: 2020-06-18 22:41
categories:
- Leetcode
tags:
- Leetcode
---

## Question
Given an array of integers that is already **sorted in ascending order**, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

You may assume that each input would have exactly one solution.

**Input:** numbers={2, 7, 11, 15}, target=9
**Output:** index1=1, index2=2

## Analysis

## Solution

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        vector<int> res;
        for (int i=0, j=numbers.size()-1; i<j;) {
            if (numbers[i]+numbers[j] > target) --j;
            else if (numbers[i]+numbers[j]<target) ++i;
            else {
                res.push_back(i+1);
                res.push_back(j+1);
                break;
            }
        }
        return res;
    }
}
```

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        i, j = 0, len(numbers)-1
        while i<j:
            sum = numbers[i] + numbers[j]
            if sum == target:
                return [i+1,j+1]
            elif sum < target:
                i+=1
            else:
                j-=1
```