# CUDA JSON Validator
Parallel algorithm for finding parentheses, locating objects and checking correctness of big JSON files with Nvidia CUDA.

## How to build?
```
nvcc common.cu main.cu -o main
```

## Sample data

* `senators.json` from [govtrack.us](https://www.govtrack.us/api/v2/role?&limit=6000) - 7.6MB, 225k lines
* `senators_short.json` from [govtrack.us](https://www.govtrack.us/api/v2/role?&limit=10) - short version of above
* `reddit_funny.json` from [reddit.com/r/funny](https://www.reddit.com/r/funny.json) - this one contains `[` and `]` symbols inside strings
* `github_events.json` from [GitHub API](https://api.github.com/events) - many levels of nested objects
