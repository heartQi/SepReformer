import os


train_mix_scp = 'tr_mix.scp'
train_s1_scp = 'tr_s1.scp'
train_s2_scp = 'tr_s2.scp'


train_mix = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/train-clean-100'
train_s1 = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/train-clean-100'
train_s2 = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/train-clean-100'


tr_mix = open(train_mix_scp,'w')
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        tr_mix.write(file+" "+root+'/'+file)
        tr_mix.write('\n')


tr_s1 = open(train_s1_scp,'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file+" "+root+'/'+file)
        tr_s1.write('\n')


tr_s2 = open(train_s2_scp,'w')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file+" "+root+'/'+file)
        tr_s2.write('\n')

test_mix_scp = 'tt_mix.scp'
test_s1_scp = 'tt_s1.scp'
test_s2_scp = 'tt_s2.scp'

test_mix = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/train-clean-360/mix_clean'
test_s1 = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/train-clean-360/s1'
test_s2 = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/train-clean-360/s2'

# test_mix = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/min/tt/mix'
# test_s1 = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/min/tt/s1'
# test_s2 = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/min/tt/s2'

tt_mix = open(test_mix_scp,'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        tt_mix.write(file+" "+root+'/'+file)
        tt_mix.write('\n')


tt_s1 = open(test_s1_scp,'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file+" "+root+'/'+file)
        tt_s1.write('\n')


tt_s2 = open(test_s2_scp,'w')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file+" "+root+'/'+file)
        tt_s2.write('\n')

cv_mix_scp = 'cv_mix.scp'
cv_s1_scp = 'cv_s1.scp'
cv_s2_scp = 'cv_s2.scp'

cv_mix = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/dev-clean/mix_both'
cv_s1 = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/dev-clean/s1'
cv_s2 = '/Users/mervin.qi/Desktop/PSE/Dataset/voicefilter_normalize/dev-clean/s2'

cv_mix_file = open(cv_mix_scp,'w')
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        cv_mix_file.write(file+" "+root+'/'+file)
        cv_mix_file.write('\n')


cv_s1_file = open(cv_s1_scp,'w')
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s1_file.write(file+" "+root+'/'+file)
        cv_s1_file.write('\n')


cv_s2_file = open(cv_s2_scp,'w')
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s2_file.write(file+" "+root+'/'+file)
        cv_s2_file.write('\n')