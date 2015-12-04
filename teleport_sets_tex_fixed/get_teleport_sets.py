import json

def get_next_date(month, day):
    month_days = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day += 1
    if day > month_days[month]:
        month += 1
        day = 1
    return month, day

def dump_teleport_sets(sets, file_name):
    print("Generating {0}".format(file_name))
    with open(file_name, 'w') as f:
        for i in range(num_users):
            serialized = json.dumps(sorted(sets[i]))
            f.write(serialized + '\n')

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

teleport_sets = [set() for i in range(num_users)]

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
            if e['type'] != 'PullRequestEvent' and e['type'] != 'PushEvent':
                continue
            user = user_map[e['actor']]
            repo = repo_map[e['repo']]
            teleport_sets[user].add(repo)
    if month == 6 and day == 30:
        dump_teleport_sets(teleport_sets, "teleport_sets-0630.txt")
    elif month == 12 and day == 31:
        dump_teleport_sets(teleport_sets, "teleport_sets-1231.txt")
    month, day = get_next_date(month, day)
