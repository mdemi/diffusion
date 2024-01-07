#!/usr/bin/python3
import sys

#'''
#image   test  https://drive.google.com/file/d/1L2yjBkzL3Ruaf8HuaMDR5PC9nETG5t_i/view?usp=drive_link
#image   train https://drive.google.com/file/d/130Lm_4K2cCclnmOEZlVIBKKPkJUn3k3f/view?usp=drive_link
#segment test  https://drive.google.com/file/d/1Q-86MIsKf7ZQxE-9_vZKVPM7FKMI15L6/view?usp=drive_link
#segment train https://drive.google.com/file/d/1YOmIt3hMc3vViH5oZ8TRnQUBrBW7rocW/view?usp=drive_link
#graph   test  https://drive.google.com/file/d/16j9m8kc6gXs_3qbB-Ta2xEPTwdQTh0r3/view?usp=drive_link
#graph   train https://drive.google.com/file/d/1iqF9IhdQS9ICKpkjnoFgTIMwa-ULtSGx/view?usp=drive_link
#'''


DATA_KEYS=dict(image   = dict(train='130Lm_4K2cCclnmOEZlVIBKKPkJUn3k3f',test='1L2yjBkzL3Ruaf8HuaMDR5PC9nETG5t_i'),
               segment = dict(train='1YOmIt3hMc3vViH5oZ8TRnQUBrBW7rocW',test='1Q-86MIsKf7ZQxE-9_vZKVPM7FKMI15L6'),
               graph   = dict(train='1iqF9IhdQS9ICKpkjnoFgTIMwa-ULtSGx',test='16j9m8kc6gXs_3qbB-Ta2xEPTwdQTh0r3'),
               )

def main(challenge,flavor):

    if not challenge in DATA_KEYS.keys() or not flavor in DATA_KEYS[challenge].keys():
        print('Usage: FLAG1 FLAG2')
        print('FLAG1:',DATA_KEYS.keys())
        print('FLAG2:',DATA_KEYS['image'].keys())
        return 1

    key = DATA_KEYS[challenge][flavor]
    
    cmd = 'gdown %s' % key
    import os
    os.system(cmd)

if __name__ == '__main__':
    import fire
    fire.Fire(main)