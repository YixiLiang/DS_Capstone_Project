# !/usr/bin/env Python
# encoding=utf-8
'''
@Project ：6501_Capstone 
@File    ：test_python.py
@Author  ：Yixi Liang
@Date    ：2023/2/9 14:46 
'''
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        path = []
        result = []
        nums.sort()
        used = [False] * len(nums)
        def backtracking(nums, used):
            if len(path) == len(nums):
                result.append(path[:])
                return
            for i in range(len(nums)):
                if used[i] == True or (i >= 1 and used[i-1] == False and nums[i] == nums[i-1]):
                    continue
                used[i] = True
                path.append(nums[i])
                backtracking(nums, used)
                path.pop()
                used[i] = False
        backtracking(nums, used)
        return result

nums = [1,1,2]
s = Solution()
result = s.permuteUnique(nums)
print(result)