#!/usr/bin/env python3

from subprocess import check_output
import os
from glob import glob
import json

check_output('pdoc --output-dir docs/api hype --force'.split())

if os.path.exists('website/sidebars.json'):
	sidebar = json.load(open('website/sidebars.json', 'r'))
	sidebar['docs']['API'] = []
else:
	sidebar = {'docs': {'API': []}}

for file in glob('docs/api/**/*.md'):
	contents = open(file, 'r').read()
	title = contents.split('\n')[0]
	id = os.path.basename(file).split('.')[0]
	with open(file, 'w') as fout:
		print('---', file=fout)
		print(f'id: {id}', file=fout)
		print(f'title: {title}', file=fout)
		print(f'sidebar_label: {title}', file=fout)
		print('---', file=fout)
		marker = '====\n'
		fout.write(contents[contents.find(marker) + len(marker):].strip())
		sidebar['docs']['API'].append(f'api/hype/{id}')

with open('website/sidebars.json', 'w') as fout:
	fout.write(json.dumps(sidebar, indent=4))