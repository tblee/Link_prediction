import json
import requests

urlstring = "https://api.github.com/search/repositories?q=created%3A<2014-01-01+pushed%3A>2014-12-31&sort=stars&per_page=100&page={0}"

repos = []
for i in range(1, 11):
    r = requests.get(urlstring.format(i))
    items = r.json()['items']
    for item in items:
        repos.append(item['full_name'])

with open("repo_list.txt", 'w') as f:
    for repo in repos:
        f.write(repo + '\n')
