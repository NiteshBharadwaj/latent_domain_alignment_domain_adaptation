import os
from PIL import Image

D = os.walk('./record_office/')
allDirectories = []
for d in D:
    allDirectories = d[1]
    break


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


for singleDirectory in allDirectories:
    D = os.walk('./record_office/' + singleDirectory)
    allFiles = ''
    baseDir1 = ''
    baseDir = ''
    for d in D:
        allFiles = d[2]
        baseDir1 = d[0] + '/' + d[1][0]
        baseDir = d[0]
        break

    before_source = ''
    before_target = ''
    after_source = ''
    after_target = ''
    print(baseDir1)
    for f in allFiles:
        if ('before_source' in f):
            before_source = Image.open(baseDir + '/' + f)
            before_source.thumbnail((200, 200), Image.ANTIALIAS)
        if ('after_source' in f):
            after_source = Image.open(baseDir + '/' + f)
            after_source.thumbnail((200, 200), Image.ANTIALIAS)
        if ('before_target' in f):
            before_target = Image.open(baseDir + '/' + f)
            before_target.thumbnail((200, 200), Image.ANTIALIAS)
        if ('after_target' in f):
            after_target = Image.open(baseDir + '/' + f)
            after_target.thumbnail((200, 200), Image.ANTIALIAS)
    try:
        v1 = get_concat_v(before_source, before_target)
        v2 = get_concat_v(after_source, after_target)
        final = get_concat_h(v1, v2)
        final.save(baseDir1 + '_tsne.png')
        print('here')
    except:
        continue
