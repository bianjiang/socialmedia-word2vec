#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s]: %(levelname)s: %(message)s')


import sys
import os
import json
import common

def transform_json_to_txt(datafolder, txtfile):


    for root, dirs, files in os.walk(os.path.abspath(datafolder)):
        for f in files:
            if (f == 'search.json'):
                continue

            with open(os.path.join(root, f), 'r', newline='', encoding='utf-8') as json_f, open(txtfile, 'a+', newline='', encoding='utf-8') as wf:
                
                logger.info('processing: [%s]'%os.path.join(root, f))

                for line in json_f:
                    #logger.info(line)

                    try:
                        tweet = json.loads(line)
                        
                        wf.write('%s '%common.sanitize_text(tweet['text']))

                    except Exception as exc:
                        #logger.warn(exc)
                        pass

if __name__ == "__main__":
    
    # handler = logging.handlers.RotatingFileHandler(
    #     'find_by_terms.log', maxBytes=10 * 1024 * 1024, backupCount=10)
    # logger.addHandler(handler)

    logger.info(sys.version)

    transform_json_to_txt('./data', './data/hpv.txt')