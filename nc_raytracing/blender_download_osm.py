from threading import Thread

import requests
import uuid
from os.path import join, dirname
import os
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent
import multiprocessing
import asyncio
from concurrent.futures import wait
import numpy as np

def splitting_a_line(lll, uuid_incl='n'):
    lll = lll.replace('(', '')
    lll = lll.replace(')', '')
    lll = lll.replace('\n', '')
    lll = lll.split(',')
    # file format: (minLon,maxLat,maxLon,minLat),percent,idx_uuid
    if uuid_incl == 'y':
        minLon, maxLat, maxLon, minLat, perc, idx_uuid = [k for k in lll]
        return float(minLon), float(maxLat), float(maxLon), float(minLat), float(perc), idx_uuid
    else:
        minLon, maxLat, maxLon, minLat, perc = [float(k) for k in lll]
        return minLon, maxLat, maxLon, minLat, perc
def consumer(queue, tqdm_size):

    pabar2 = tqdm(total=tqdm_size, position=0, desc="Saving", leave=True)
    res = []
    while True:
        # get a unit of work
        item = queue.get(block=True)

        pabar2.update(1)
        # check for stop
        res.append(item)

        if len(res) % 10 == 0 or  item is None:
            file1 = open(BASE_PATH + RES_FILE_NAME, 'a')
            file1.writelines(res)
            file1.close()
            res.clear()

        if item is None:
            break





def producer_download_osm(queue, BLENDER_OSM_DOWNLOAD_PATH, line, idx):


    minLon, maxLat, maxLon, minLat, perc = splitting_a_line(lll=line, uuid_incl='n')

    url = 'http://tc319-srv1.egr.duke.edu:23412/api/map?bbox={:f},{:f},{:f},{:f}'.format(minLon, minLat, maxLon, maxLat)
    response = requests.get(url)
    osm_text = response.text
    idx_uuid = str(idx) + '_' + str(uuid.uuid4())

    # save osm:
    f_ptr_osm = open(BLENDER_OSM_DOWNLOAD_PATH + idx_uuid + '.osm', 'w')
    f_ptr_osm.writelines(osm_text)
    f_ptr_osm.close()
    temp_arr = [minLon, maxLat, maxLon, minLat, perc, idx_uuid]
    temp_arr = [str(t) for t in temp_arr]
    # save line in res:
    res = '(' + ','.join(temp_arr[0:4]) + '),' + ','.join(temp_arr[-2:]) + '\n'
    queue.put(res)

def producer(batch_lines, batch_idxs, queue, BLENDER_OSM_DOWNLOAD_PATH):

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for line, idx in zip(batch_lines, batch_idxs):
            executor.submit(producer_download_osm, queue, BLENDER_OSM_DOWNLOAD_PATH, line, idx)
        # executor.submit(compute_building_to_land_ration, job[0], job[1], queue, to4326) for job in batch
        # wait for tasks to cwait(futures)omplete
        #_ = wait(futures)

if __name__ == '__main__':
    load_dotenv(join(dirname(__file__), '.env'))

    BASE_PATH = os.environ.get('BASE_PATH')
    # BLENDER_PATH should be the path I built, since the things are enabled
    BLENDER_OSM_DOWNLOAD_PATH = os.environ.get('BLENDER_OSM_DOWNLOAD_PATH')
    RES_FILE_NAME = os.environ.get('RES_FILE_NAME')
    # IDX_STOP = 10

    # http://tc319-srv1.egr.duke.edu:23412/api/map?bbox=-100.702392578125,25.30952262878418,-100.69202423095703,25.318172454833984
    # this: -100.702389,25.318173,-100.692025,25.309523
    # is:    minLonOut, maxLatOut, maxLonOut, minLatOut
    # so, the server request follows minLon, minLat, maxLon, maxLat

    f_ptr_res = open(BASE_PATH + 'res3_srv1_whole_us_filtered.txt', 'r')
    lines = f_ptr_res.readlines()
    f_ptr_res.close()

    m = multiprocessing.Manager()
    queue = m.Queue()
    futures = []
    try:
        consumer_process = Thread(target=consumer, args=(queue, len(lines)))
        consumer_process.start()
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            batch_size = 100
            idxs = np.arange(len(lines))
            for i in range(0, len(lines), batch_size):
                batch_lines = lines[i:i + batch_size]  # the result might be shorter than batchsize at the end
                batch_idxs = idxs[i:i + batch_size]  # the result might be shorter than batchsize at the end
                # do stuff with batch
                # for job in job_queue:
                a_result = executor.submit(producer, batch_lines, batch_idxs, queue, BLENDER_OSM_DOWNLOAD_PATH)
            # for idx, line in enumerate(lines):
            #     futures.append(executor.submit(producer_download_osm, queue, BLENDER_OSM_DOWNLOAD_PATH, line, idx))
    except KeyboardInterrupt:
        for job in futures:
            job.cancel()
    finally:
        queue.put(None)
        wait(futures)
        consumer_process.join()