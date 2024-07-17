from cut64window import cut64window, filter_img, update_mask
from Lesps_old import run_lesps
from DNA import run_dna
from DNA_test import test_dna
# from UIU_train import run_trainUIU
# from test_ICPR import run_test_ICPR







if __name__ == '__main__':
    cut64window()
    run_lesps()
    print('filtering and updating...')
    filter_img()
    update_mask()
    run_dna()
    test_dna()
    run_trainUIU()
    run_test_ICPR()


