import json
import sys
from logging import error

def test_aer_raw(squad, aer_list):
    error_list = []
    for i, aer in enumerate(aer_list):
        ae_txt, ae_st = aer
        ae_ed = ae_st + len(ae_txt)
        span = squad['context'][ae_st: ae_ed]
        
        if ae_txt != span:
            error_list.append(i)
            print(i)
            print(ae_txt)
            print(span)
            print('-------------------------')
    
    if len(error_list) == 0:
        print('\n* AER-span matching test is passed *\n')
    sys.exit()