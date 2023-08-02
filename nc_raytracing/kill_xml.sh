#!/bin/bash
kill $(ps aux | grep 'xml_to' | awk '{print $2}')
