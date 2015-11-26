import json

contrib_threshold = 1
interact_threshold = 1

num_repos = None
num_users = None

def get_next_date(month, day):
    month_days = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day += 1
    if day > month_days[month]:
        month += 1
        day = 1
    return month, day

def dump_edges(nodes, file_name):
    print("Generating {0}".format(file_name))
    with open(file_name, 'w') as f:
        for i in range(num_repos):
            for j in range(i+1, num_repos):
                common_contribs = len(nodes[i]['contrib'].intersection(nodes[j]['contrib']))
                common_interacts = len(nodes[i]['interact'].intersection(nodes[j]['interact']))
                if common_contribs >= contrib_threshold and common_interacts >= interact_threshold:
                    f.write("{0},{1},{2},{3}\n".format(i, j, common_contribs, common_interacts))

repos = []
with open('repo_list.txt') as f:
    repos = [line.rstrip() for line in f.readlines()]
num_repos = len(repos)
repo_map = {repos[i]: i for i in range(num_repos)}

users = []
with open('user_list.txt') as f:
    users = [line.rstrip() for line in f.readlines()]
num_users = len(users)
user_map = {users[i]: i for i in range(num_users)}

nodes = []
for i in range(num_repos):
    nodes.append({'contrib': set(), 'interact': set()})

month = 1
day = 1
while month < 13:
    if day == 1:
        print("Reading data for month {0}".format(month))
    with open("data/{0:02}{1:02}-all.json".format(month, day)) as f:
        for line in f:
            e = json.loads(line.rstrip())
            if e['actor'] not in user_map:
                continue
            user = user_map[e['actor']]
            repo = repo_map[e['repo']]
            nodes[repo]['interact'].add(user)
            if e['type'] == 'PullRequestEvent' or e['type'] == 'PushEvent':
                nodes[repo]['contrib'].add(user)
    if month == 6 and day == 30:
        dump_edges(nodes, "snapshot-0630.txt")
    elif month == 12 and day == 31:
        dump_edges(nodes, "snapshot-1231.txt")
    month, day = get_next_date(month, day)
