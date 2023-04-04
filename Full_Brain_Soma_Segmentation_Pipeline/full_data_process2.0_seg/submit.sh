#!/bin/bash

curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer $1" \
     -X POST http://192.168.16.72:80/rest-server/api/v1/jobs \
     -d @$2
