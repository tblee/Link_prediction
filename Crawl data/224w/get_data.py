import gzip
import json
import requests

urlstring = "http://data.githubarchive.org/2014-{0:02}-{1:02}-{2}.json.gz"
f_all_string = "data/{0:02}{1:02}-all.json"
f_contribs_string = "data/{0:02}{1:02}-contribs.json"

concerned_events = set(['ForkEvent', 'IssueCommentEvent', 'IssuesEvent', 'PullRequestEvent', 'PushEvent', 'WatchEvent'])

def get_next_time(month, day, time):
    month_days = [None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    time += 1
    if time <= 23:
        return month, day, time
    day += 1
    if day <= month_days[month]:
        return month, day, 0
    return month+1, 1, 0

repos = set()
with open('repo_list.txt') as f:
    for line in f:
        repos.add(line.rstrip())

month = 1
day = 1
time = 0
while month < 13:
    if time == 0:
        print("Processing 2014/{0:02}/{1:02}".format(month, day))
        f_all = open(f_all_string.format(month, day), 'w')
        f_contribs = open(f_contribs_string.format(month, day), 'w')
    
    url = urlstring.format(month, day, time)
    r = requests.get(url)
    try:
        lines = gzip.decompress(r.content).decode().split('\n')
    except Exception:
        print("Could not process time {0}".format(time))
        month, day, time = get_next_time(month, day, time)
        continue
    if lines[-1] == '':
        lines.pop()
    
    for line in lines:
        e = json.loads(line)
        e_type = e['type']
        if e_type not in concerned_events or 'repository' not in e:
            continue
        repo = e['repository']['owner'] + '/' + e['repository']['name']
        if repo not in repos:
            continue
        res = {'type': e_type, 'actor': e['actor'], 'repo': repo}
        serialized = json.dumps(res)
        f_all.write(serialized + '\n')
        if e_type == 'PullRequestEvent' or e_type == 'PushEvent':
            f_contribs.write(serialized + '\n')
    
    if time == 23:
        f_all.close()
        f_contribs.close()
    month, day, time = get_next_time(month, day, time)
