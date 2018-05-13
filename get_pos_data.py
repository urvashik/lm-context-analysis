import sys

if len(sys.argv) < 2:
    print('Need the dataset name!')
    exit(0)

for split in ['train', 'valid', 'test']:
    with open('data/'+sys.argv[1]+'/'+split+'.txt', 'r') as f1, open(
              'data/'+sys.argv[1]+'_pos_only/'+split+'.txt', 'r') as f2, open(
              'data/'+sys.argv[1]+'_pos/'+split+'.txt', 'w') as fout:

        for i, (line, pline) in enumerate(zip(f1,f2)):
            if line.strip().split(' ')[0] == '': # empty lines in wiki
                fout.write(line)
                continue

            line = line.strip().split(' ')
            pline = pline.strip().split(' ')

            line = [w+'_'+p for w, p in zip(line, pline)]
            fout.write(' '.join(line)+' \n')
