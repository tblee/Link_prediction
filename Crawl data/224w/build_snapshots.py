import json
from collections import OrderedDict

num_repos = 100

def get_next_date(month, day):
    month_days = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day += 1
    if day > month_days[month]:
        month += 1
        day = 1
    return month, day

def dump_nodes(nodes, file_name):
    with open(file_name, 'w') as f:
        for node in nodes:
            f.write(json.dumps(node) + '\n')

repos = []
with open('repo_list.txt') as f:
    repos = [line.rstrip() for line in f.readlines()]
repo_map = {repos[i]: i for i in range(len(repos))}

users = []
with open('user_list.txt') as f:
    users = [line.rstrip() for line in f.readlines()]
user_map = {users[i]: i for i in range(len(users))}

nodes = []
for i in range(len(users)):
    nodes.append(OrderedDict([('id', i), ('contrib', [0]*num_repos), ('interact', [0]*num_repos)]))

month = 1
day = 1
while month < 13:
    with open("data/{0:02}{1:02}-all.json".format(month, day)) as f:
        for line in f:
            e = json.loads(line.rstrip())
            if e['actor'] not in user_map:
                continue
            user = user_map[e['actor']]
            repo = repo_map[e['repo']]
            nodes[user]['interact'][repo] = 1
            if e['type'] == 'PullRequestEvent' or e['type'] == 'PushEvent':
                nodes[user]['contrib'][repo] = 1
    if month == 6 and day == 30:
        dump_nodes(nodes, "snapshot-0630.json")
    elif month == 12 and day == 31:
        dump_nodes(nodes, "snapshot-1231.json")
    month, day = get_next_date(month, day)

count = 0
for node in nodes:
    if sum(node['interact']) - sum(node['contrib']) > 1:
        count += 1
print(count)
