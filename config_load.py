# json file 로드하기 위한 코드입니다.

import json
with open('/tf/nasw/config.json') as f:
    config=json.load(f)