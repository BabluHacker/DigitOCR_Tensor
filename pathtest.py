import os

paths=['/home/mehedi/AI/MAIN/test']
for path in paths:
    for root,directives,filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(root, filename)
            t = fullpath.rindex('/')
            print fullpath
            print fullpath[0:t]
dirc='/home/mehedi/AI/MAIN/test'
for root,directives,filenames in os.walk(dirc):
    for filename in filenames:
            nfilename=filename+"_22"
            os.rename(filename,nfilename)