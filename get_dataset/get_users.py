import json

def get_next_date(month, day):
    month_days = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day += 1
    if day > month_days[month]:
        month += 1
        day = 1
    return month, day

users = {}

month = 1
day = 1
while month < 13:
    with open("data/{0:02}{1:02}-contribs.json".format(month, day)) as f:
        for line in f:
            e = json.loads(line.rstrip())
            actor = e['actor']
            if actor not in users:
                users[actor] = set()
            users[actor].add(e['repo'])
        month, day = get_next_date(month, day)

print("Total number of users: {0}".format(len(users)))
with open('user_list.txt', 'w') as f:
    for user in users:
        f.write(user + '\n')
