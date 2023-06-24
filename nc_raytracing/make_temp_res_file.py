import uuid
from blender_test_wraper import BASE_PATH, splitting_a_line


def temp_file_gen(f_orig_ptr, f_len=10):
    """
    This file is used to append an idx_uuid str to every line
    """
    lines_inner = f_orig_ptr.readlines()
    f_new_ptr = open(BASE_PATH + 'res3_srv1_whole_us_filtered_new.txt', 'w')
    for idx, lll in enumerate(lines_inner):
        temp_arr = list(splitting_a_line(lll=lll, uuid_incl='n'))
        temp_arr.append(str(idx) + '_' + str(uuid.uuid4()))
        temp_arr = [str(t) for t in temp_arr]
        f_new_ptr.write('(' + ','.join(temp_arr[0:4]) + '),' + ','.join(temp_arr[-2:]) + '\n')
        if idx >= f_len - 1:
            break
    f_new_ptr.close()
    return


if __name__ == '__main__':
    with open(BASE_PATH + 'res3_srv1_whole_us_filtered.txt', 'r') as loc_fPtr:
        temp_file_gen(loc_fPtr)
