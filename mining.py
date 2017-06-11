#!/usr/bin/python3
import tweepy
import re

consumer_key = '...'
consumer_secret = '...'
access_token = '...'
access_token_secret = '...'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# flag 0: Hilary支持者
# flag 1: Trump支持者
# flag 2: 中立
tags = [
    (0, 'TrumpObstructed'),
    (0, 'ComeyIsAPatriot'),
    (0, 'TrumpRussia'),
    (0, 'russiagate'),
    (0, 'ImpeachTrump'),
    (0, 'TrumpLeaks'),
    (1, 'trumpwon'),
    (1, 'AmericaFirst'),
    (1, 'SethRich'),
    (1, 'ClintonEmails'),
    (1, 'ClintonBodyCount'),
    (1, 'LockHerUp'),
    (1, 'MAGA'),
    (1, 'MakeAmericaGreatAgain'),
    (1, 'trumptrain'),
    (2, 'november8'),
    (2, 'nov8'),
    (2, 'Election2016'),
    (2, '2016Election'),
    (2, 'americaelection'),
    (2, 'usaelection'),
    (2, 'WhiteHouse')
]

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F6FF"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U0001F900-\U0001F9FF"
                           "]+", flags=re.UNICODE)

MAX_NUM_WITH_TAG = 1000

# mining data from twitter
for flag, tag in tags:
    counter = 0
    tag = tag.lower()
    search_result = tweepy.Cursor(api.search,
        q='#'+tag,
        count=100,
        result_type='recent',
        include_entities=True,
        lang='en'
        ).items()
    f = open(str(flag) + '_' + tag + '.csv', 'w')
    for tweet in search_result:
        if counter == MAX_NUM_WITH_TAG:
            break
        elif (not tweet.retweeted) and ('RT @' not in tweet.text) and ('via @' not in tweet.text):
            counter += 1
            # remove newlines
            words = tweet.text.replace('\n', ' ').replace('\r', '').split()
            # remove @ tag, http(s) urls, &amp;
            words = [w for w in words if 
                (w[0] != '@') and 
                (not w.startswith('http')) and 
                (not w.startswith('&'))]
            # remove emoji
            words = emoji_pattern.sub(r'', ' '.join(words))
            # remove leading and trailing spaces
            words = words.strip()
            if words == '':
                counter -= 1
            else:
                if flag == 2:
                    f.write(words + '\n')
                else:
                    f.write(str(flag) + ',' + words + '\n')
                print(
                    '\x1b[1;33;40m' + str(flag) + '\x1b[0m' + ' ' +
                    '\x1b[1;33;41m' + tag + '\x1b[0m' + ' ' +
                    '\x1b[0;33;45m' + str(counter) + '\x1b[0m' + ' ' +
                    words)
    print('============================================')
    print(str(flag) + ' ' + tag + ' ' + str(counter))
    print('============================================')
    f.close()

# merge results
found_0 = set([])
found_1 = set([])
found_2 = set([])
fout_0 = open('0.csv', 'w')
fout_1 = open('1.csv', 'w')
fout_2 = open('2.csv', 'w')

for flag, tag in tags:
    fin = open(str(flag) + '_' + tag.lower() + '.csv', 'r')
    for line in fin:
        if flag == 0 and line not in found_0:
            found_0.add(line)
            fout_0.write(line)
        elif flag == 1 and line not in found_1:
            found_1.add(line)
            fout_1.write(line)
        elif flag == 2 and line not in found_2:
            found_2.add(line)
            fout_2.write(line)
