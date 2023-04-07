#!/usr/bin/env python2

from subprocess import Popen, PIPE

import os
import urllib2
import sys

utilDir = os.path.dirname(os.path.realpath(__file__))

ignores = ['localhost', '127.0.0.1', 'your-server', 'docker-ip',
           'ghbtns', 'sphinx-doc']


def ignoreURL(url):
    for ignore in ignores:
        if ignore in url:
            return True
    return False

hdr = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'
}

dirs = ['api-docs', 'batch-represent', 'docs', 'evaluation',
        'openface', 'training', 'util']
dirs = [os.path.join(utilDir, '..', d) for d in dirs]
cmd = ['grep', '-I', '--no-filename',
       '-o', '\(http\|https\)://[^"\')}`<> ]*',
       '-R'] + dirs + \
    ['--exclude-dir=_build']

p = Popen(cmd, stdout=PIPE)
out = p.communicate()[0]
urls = set(out.split())

badURLs = []
for url in urls:
    if not ignoreURL(url):
        if url.endswith('.'):
            url = url[:-1]
        print("+ {}".format(url))
        try:
            req = urllib2.Request(url, headers=hdr)
            resp = urllib2.urlopen(req)
        except Exception as e:
            print("  + Error:\n\n")
            print(e)
            print("\n\n")
            badURLs.append(url)

print('\nFound {} bad of {} URLs'.format(len(badURLs), len(urls)))
if len(badURLs) > 0:
    print("\n\n=== Bad URLs.\n")
    for url in badURLs:
        print("+ {}".format(url))
    sys.exit(-1)
