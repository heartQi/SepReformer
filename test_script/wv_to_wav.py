"""
# example:
# 11-1.1/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and
# stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav
"""
import os

# the root dir for WSJ0 corpus
root_dir = "/Users/mervin.qi/Desktop/PSE/Dataset/csr_1"

# the disc number
disc_dir = []
for list_disc in os.listdir(root_dir):
    if list_disc not in ["text", "11-13.1"]: #doc file and 11-13.1 file do not contain .wv files
        # the data dir for each disc
        disc_dir.append(os.path.join(root_dir, list_disc, "wsj0"))

my_path = "/Users/mervin.qi/Desktop/PSE/Dataset/wsj0_LDC93S6A."
if not os.path.exists(my_path):
    os.mkdir(my_path)
# # the sub_data dir for each disc
for i, list_sub_data in enumerate(disc_dir):
    for sub_data_dir in os.listdir(list_sub_data):
        if (not sub_data_dir.startswith("si")) and (not sub_data_dir.startswith("sd")):
            continue
        s_dir = os.path.join(my_path, sub_data_dir)
        if not os.path.exists(s_dir):
            os.mkdir(s_dir)
        if sub_data_dir[0][0] == 's':
            datatype_dir = os.path.join(list_sub_data, sub_data_dir)
            for list_spk in os.listdir(datatype_dir):
                spk_dir = os.path.join(s_dir, list_spk)
                spk_dir_abs = os.path.join(datatype_dir, list_spk)
                if not os.path.exists(spk_dir):
                    os.mkdir(spk_dir)
                for wv_file in os.listdir(spk_dir_abs):
                    if (not wv_file.endswith('.wv1')) and (not wv_file.endswith('.wv2')):
                        continue
                    speech_dir = os.path.join(spk_dir_abs, wv_file)
                    if wv_file.split('.')[1] == "wv1":
                        target_name = wv_file.split(sep='.')[0] + '.wav'
                    elif wv_file.split('.')[1] == 'wv2':
                        target_name = wv_file.split(sep='.')[0] + '_1.wav'

                    target_dir = spk_dir + '\\' + target_name
                    # rif == wav
                    cmd = "/Users/mervin.qi/Desktop/PSE/Dataset/sph2pipe_v2.5/sph2pipe -f wav " + speech_dir + " " + target_dir
                    os.system(cmd)
